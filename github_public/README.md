# Chaos CUDA Kernels — 混沌神经网络硬件加速内核

[![CUDA](https://img.shields.io/badge/CUDA-13.2-green)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell](https://img.shields.io/badge/Arch-sm_120-black)](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-5070-ti/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Triton](https://img.shields.io/badge/Triton-3.6-orange)](https://triton-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

手写CUDA内核 + CUDA Driver API加载 + PyTorch优化器接口。

> 本仓库为通用加速工具集。完整训练系统、WebUI、模型架构等为商业产品，未开源。

## 目录

```
├── chaos_adam_blockwise.cu/.ptx   # 融合Adam 4-bit内核
├── chaos_adam_fp8.cu/.ptx         # 融合Adam FP8 Tensor Core内核
├── henon_perturb.cu/.ptx          # 并行Henon扰动内核
├── dwc_fused.cu/.ptx              # DWC融合前向内核
├── ptx_loader.py                  # CUDA Driver API加载器
├── API.py                         # 统一API (工厂模式)
├── cuda_kernel_optimizer.py       # 4-bit PyTorch优化器
├── fp8_cuda_graph_trainer.py      # FP8 + CUDA Graph训练器
├── chaos_triton_fused.py          # Triton回退版内核
├── README.md
└── .gitignore
```

## 已编译内核 (Blackwell GB205)

| 内核 | 精度 | 硬件单元 |
|------|------|----------|
| `chaos_adam_blockwise.ptx` | 4-bit uint8 | SFU / Warp Shuffle / TMEM |
| `chaos_adam_fp8.ptx` | FP8 E4M3 | Tensor Core / Constant Cache / fmaf |
| `henon_perturb.ptx` | FP32 | float4 向量化 / fmaf |
| `dwc_fused.ptx` | FP32 | Shared Memory / SFU |

### 编译

```bash
nvcc -O3 -use_fast_math --restrict -arch=sm_120 \
     -maxrregcount 64 -Xptxas -dlcm=ca \
     -ptx xxx.cu -o xxx.ptx
```

## 使用

```python
import sys, os
sys.path.insert(0, 'cuda_kernels')
from API import create_optimizer, print_status

# FP8 Tensor Core 优化器
opt = create_optimizer('fp8', params=model.parameters(), lr=1e-3)

# 4-bit 极省显存优化器
opt = create_optimizer('4bit', params=model.parameters())

# 训练循环中使用
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    opt.step()
    opt.zero_grad()

# 查看状态
print_status()
```

## 硬件要求

- NVIDIA RTX 5070 Ti (Blackwell GB205, `sm_120`)
- CUDA 13.2+
- VS2026 (编译 .cu)
- Triton 3.6+ (Triton版)

## License

MIT License — 详见 [LICENSE](LICENSE)

---

*核心混沌神经网络模型、涌现引擎、WebUI 训练平台为商业产品，未包含在本仓库。*
