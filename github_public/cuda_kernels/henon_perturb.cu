/*
 * henon_perturb.cu — Henon混沌扰动 (Blackwell全加速)
 * =====================================================
 * 硬件使用:
 *   - __constant__: Henon a/b 系数 (Constant Cache)
 *   - float4 向量化: 128-bit coalesced 读写 (L2 NVCache 最优带宽)
 *   - SFU: fmaf 硬件融合乘加
 *   - __launch_bounds__: 最大化SM占用
 *
 * 编译:
 *   nvcc -O3 -use_fast_math --restrict -arch=sm_120 \
 *        -maxrregcount 48 -Xptxas -dlcm=ca \
 *        -ptx henon_perturb.cu -o henon_perturb.ptx
 */

#include <cuda_runtime.h>

__constant__ float c_henon_a = 1.4f;
__constant__ float c_henon_b = 0.3f;

__launch_bounds__(256, 8)
extern "C" __global__ void henon_perturb_kernel(
    float* grad,
    float* henon_x,
    float* henon_y,
    int n_elements,
    float lambda,
    float lyapunov_factor
) {
    // ─── 向量化索引 (一次处理4个float = 128-bit coalesced) ───
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // 标量处理边界
    for (int i = base; i < base + 4 && i < n_elements; i++) {
        float x = henon_x[i];
        float y = henon_y[i];
        
        // fmaf: 硬件融合乘加 (SFU)
        float x_new = 1.0f - fmaf(c_henon_a, x * x, -y);  // = 1 - 1.4*x^2 + y
        float y_new = c_henon_b * x;
        
        henon_x[i] = x_new;
        henon_y[i] = y_new;
        
        // 混沌扰动 (fmaf)
        grad[i] = fmaf(x_new, lambda * lyapunov_factor, grad[i]);
    }
}

extern "C" void launch_henon_perturb(
    float* grad,
    float* henon_x,
    float* henon_y,
    int n_elements,
    float lambda,
    float lyapunov_factor,
    cudaStream_t stream
) {
    int threads = 256;
    // 每个线程处理4个元素
    int elems_per_block = threads * 4;
    int blocks = (n_elements + elems_per_block - 1) / elems_per_block;
    henon_perturb_kernel<<<blocks, threads, 0, stream>>>(
        grad, henon_x, henon_y, n_elements, lambda, lyapunov_factor
    );
}
