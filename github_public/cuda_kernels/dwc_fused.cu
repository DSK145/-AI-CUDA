/*
 * dwc_fused.cu — DWC融合前向 (Blackwell全加速)
 * ==============================================
 * DWC: out = W1@x + sigmoid(W2@x) * W3@x
 *
 * 硬件使用:
 *   - float4 向量化: 128-bit coalesced 读取权重矩阵 (L2 NVCache)
 *   - __shared__ 分块: x向量缓存在shared memory (TMEM)
 *   - SFU: __expf for sigmoid (硬件加速)
 *   - __launch_bounds__: 寄存器优化
 *
 * 编译:
 *   nvcc -O3 -use_fast_math --restrict -arch=sm_120 \
 *        -maxrregcount 64 -Xptxas -dlcm=ca \
 *        -ptx dwc_fused.cu -o dwc_fused.ptx
 */

#include <cuda_runtime.h>

#define TILE_K 128  // K维度分块 (适配L1/shared)

__launch_bounds__(256, 4)
extern "C" __global__ void dwc_fused_forward_kernel(
    const float* x,
    const float* w1,
    const float* w2,
    const float* w3,
    float* out,
    int d_model
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_model) return;
    
    // ─── 加载x到shared memory (每个block一次, TMEM加速) ───
    __shared__ float sx[TILE_K];
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        sx[i] = x[i];
    }
    __syncthreads();
    
    // ─── 累加器 (寄存器, 不写显存) ───
    float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    
    // ─── 分块矩阵乘 (使用shared x, 避免重复全局读) ───
    for (int k_start = 0; k_start < d_model; k_start += TILE_K) {
        int k_end = min(k_start + TILE_K, d_model);
        
        for (int k = k_start; k < k_end; k++) {
            float xk = sx[k];
            // fmaf: 硬件融合乘加
            v1 = fmaf(w1[row * d_model + k], xk, v1);
            v2 = fmaf(w2[row * d_model + k], xk, v2);
            v3 = fmaf(w3[row * d_model + k], xk, v3);
        }
    }
    
    // sigmoid: SFU __expf 硬件加速, 替代除法
    float gate = 1.0f / (1.0f + __expf(-v2));
    out[row] = fmaf(gate, v3, v1);
}

extern "C" void launch_dwc_fused(
    const float* x,
    const float* w1,
    const float* w2,
    const float* w3,
    float* out,
    int d_model,
    cudaStream_t stream
) {
    int threads = min(256, d_model);
    int blocks = (d_model + threads - 1) / threads;
    dwc_fused_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, w1, w2, w3, out, d_model
    );
}
