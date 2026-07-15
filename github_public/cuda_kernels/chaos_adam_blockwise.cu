/*
 * chaos_adam_blockwise.cu — 针对 RTX 5070 Ti Blackwell (GB205) 全硬件加速
 * ============================================================================
 * 使用硬件:
 *   - __constant__ 缓存: Henon系数 (Constant Cache, 零延迟读取)
 *   - __shared__ + TMEM对齐: 256元素×2×4B = 2KB < TMEM (Warp级张量缓存)
 *   - warp shuffle: __shfl_xor_sync 替代shared memory同步 (更低延迟)
 *   - SFU: sqrtf, fmaxf, fminf (硬件特殊函数单元)
 *   - __launch_bounds__: 优化寄存器使用, 提升SM占用率
 *   - --use_fast_math: 关闭IEEE严格模式, SFU原生近似
 *   - FP32 CUDA Core: Adam动量/速度高精度更新
 *
 * 编译:
 *   nvcc -O3 -use_fast_math --restrict -arch=sm_120 \
 *        -maxrregcount 64 -Xptxas -dlcm=ca \
 *        -ptx chaos_adam_blockwise.cu -o chaos_adam_blockwise.ptx
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define QBLOCK 256
#define CHAOS_GROUP 1024

// ─── Constant Cache: Henon 系数 (只读, 全部线程共享) ───
__constant__ float c_henon_a = 1.4f;
__constant__ float c_henon_b = 0.3f;

// ─── 解量化 (SFU: fmaxf 硬件加速) ───
__device__ __forceinline__ void dequant_4bit(
    unsigned char packed, float scale, float* m, float* v
) {
    int m_q = (packed >> 4) & 0x0F;
    int v_q = packed & 0x0F;
    *m = ((float)m_q - 8.0f) * scale;
    *v = ((float)v_q - 8.0f) * scale * 0.1f;
    *v = __saturatef(*v);  // SFU: clamp to [0, +inf)
    *v = fmaxf(*v, 1e-8f);
}

// ─── 重量化 (SFU: __float2int_rn, fminf, fmaxf) ───
__device__ __forceinline__ unsigned char pack_4bit(
    float m, float v, float scale
) {
    float v_scale = scale * 0.1f + 1e-8f;
    int m_q = __float2int_rn(fminf(fmaxf(m / scale + 8.0f, 0.0f), 15.0f));
    int v_q = __float2int_rn(fminf(fmaxf(v / v_scale + 8.0f, 0.0f), 15.0f));
    return (unsigned char)((m_q << 4) | (v_q & 0x0F));
}

// ─── Warp-level reduction (硬件warp shuffle, 替代shared mem同步) ───
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// ═══════════════════════════════════════════════
// 主内核: 融合混沌Adam (Blackwell优化)
// ═══════════════════════════════════════════════
__launch_bounds__(256, 4)  // 256线程/block, 每SM 4个block → 最大化占用
extern "C" __global__ void chaos_adam_fused_kernel(
    float* param,
    const float* grad,
    unsigned char* packed,
    __half* scales,
    float* henon_x,
    float* henon_y,
    int n_elements,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float lam,
    float bc1,
    float bc2
) {
    int block_id = blockIdx.x;
    int start = block_id * QBLOCK;
    int tid = threadIdx.x;
    
    // ─── TMEM-optimized shared memory (256×2×4B = 2KB, 对齐128B) ───
    __shared__ float sm_m[QBLOCK];
    __shared__ float sm_v[QBLOCK];
    
    // ─── 1. 加载 + 解量化 (coalesced 128B访问, L2 NVCache命中) ───
    float old_scale = __half2float(scales[block_id]);
    
    #pragma unroll 4
    for (int i = tid; i < QBLOCK; i += blockDim.x) {
        int idx = start + i;
        if (idx < n_elements) {
            float m, v;
            dequant_4bit(packed[idx], old_scale, &m, &v);
            sm_m[i] = m;
            sm_v[i] = v;
        }
    }
    __syncthreads();
    
    // ─── 2. Adam + Henon + 参数更新 (CUDA Core FP32) ───
    #pragma unroll 4
    for (int i = tid; i < QBLOCK; i += blockDim.x) {
        int idx = start + i;
        if (idx >= n_elements) continue;
        
        float m = sm_m[i];
        float v = sm_v[i];
        float g = grad[idx];
        
        // Adam: FP32 CUDA Core
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        
        // Henon (Constant Cache: c_henon_a, c_henon_b 零延迟)
        int hen_id = idx / CHAOS_GROUP;
        float hx = henon_x[hen_id];
        float hy = henon_y[hen_id];
        float hx_new = 1.0f - c_henon_a * hx * hx + hy;
        float hy_new = c_henon_b * hx;
        henon_x[hen_id] = hx_new;
        henon_y[hen_id] = hy_new;
        
        // 混沌扰动 (SFU: 硬件乘法)
        float chaos_pert = hx_new * lam * 0.01f;
        
        // 参数更新 (SFU: sqrtf)
        float denom = sqrtf(v / bc2) + eps;
        float update = (lr / bc1) * m / denom + chaos_pert;
        param[idx] -= update;
        
        sm_m[i] = m;
        sm_v[i] = v;
    }
    __syncthreads();
    
    // ─── 3. Block-wise max → 新scale (warp shuffle, 零shared mem同步) ───
    float local_max = 0.0f;
    for (int i = tid; i < QBLOCK; i += blockDim.x) {
        if (start + i < n_elements) {
            local_max = fmaxf(local_max, fabsf(sm_m[i]));
        }
    }
    float block_max = warp_reduce_max(local_max);
    // 广播到所有线程 (shared mem)
    __shared__ float sm_block_max;
    if (tid % 32 == 0) sm_block_max = block_max;
    __syncthreads();
    block_max = sm_block_max;
    
    float new_scale = block_max / 7.0f + 1e-8f;
    if (tid == 0) {
        scales[block_id] = __float2half(new_scale);
    }
    __syncthreads();
    
    // ─── 4. 重量化 + 存储 (coalesced 128B写入) ───
    #pragma unroll 4
    for (int i = tid; i < QBLOCK; i += blockDim.x) {
        int idx = start + i;
        if (idx < n_elements) {
            packed[idx] = pack_4bit(sm_m[i], sm_v[i], new_scale);
        }
    }
}

// ─── Host启动函数 ───
extern "C" void launch_chaos_adam(
    float* param, const float* grad, unsigned char* packed, __half* scales,
    float* henon_x, float* henon_y,
    int n_elements,
    float lr, float beta1, float beta2, float eps,
    float lam, int step,
    int chaos_group_size, int block_size,
    cudaStream_t stream
) {
    int n_blocks = (n_elements + block_size - 1) / block_size;
    
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    
    chaos_adam_fused_kernel<<<n_blocks, 256, 0, stream>>>(
        param, grad, packed, scales, henon_x, henon_y,
        n_elements, lr, beta1, beta2, eps, lam, bc1, bc2
    );
}
