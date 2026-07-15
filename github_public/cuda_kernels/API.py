#!/usr/bin/env python3
"""
cuda_kernels/API — 混沌CUDA内核统一API
=======================================
导入方式:
  import sys, os
  sys.path.insert(0, os.path.join(ROOT, 'cuda_kernels'))
  from API import *

提供:
  ┌─ 原生PTX内核 ─────────────────┐
  │ chaos_adam_4bit()    4-bit Adam │
  │ chaos_adam_fp8()     FP8  Adam  │
  │ henon_perturb()      Henon扰动  │
  │ dwc_fused()          DWC融合    │
  ├─ 优化器 ───────────────────────┤
  │ NativeCUDAOptimizer  4-bit优化器│
  │ FP8ChaosOptimizer    FP8优化器  │
  ├─ 训练器 ───────────────────────┤
  │ CUDAGraphTrainer     Graph训练  │
  └─ 工具 ────────────────────────┘
  │ sync()              CUDA同步   │
  │ init_cuda()         初始化     │
  │ get_status()        状态报告   │
  └────────────────────────────────┘
"""
import os as _os
import sys as _sys

# 确保当前目录在路径中
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import torch
import triton

# ═══════════════════════════════════════════
# Part 1: PTX 原生内核 (CUDA Driver API)
# ═══════════════════════════════════════════

from ptx_loader import (
    # 初始化
    init as _ptx_init,
    sync,
    
    # 内核
    chaos_adam_cuda as chaos_adam_4bit,
    chaos_adam_fp8_cuda as chaos_adam_fp8,
    henon_perturb_cuda as henon_perturb,
    dwc_fused_cuda as dwc_fused,
    
    # 底层
    load_ptx,
    get_kernel,
    launch_kernel,
)

# ═══════════════════════════════════════════
# Part 2: Triton内核 (备用, 无编译器时)
# ═══════════════════════════════════════════

try:
    from chaos_triton_fused import (
        ChaosTritonOptimizer,
        triton_chaos_adam_step,
    )
    _has_triton_kernels = True
except ImportError:
    ChaosTritonOptimizer = None
    triton_chaos_adam_step = None
    _has_triton_kernels = False

# ═══════════════════════════════════════════
# Part 3: 优化器 (PyTorch Optimizer接口)
# ═══════════════════════════════════════════

from cuda_kernel_optimizer import NativeCUDAOptimizer
from fp8_cuda_graph_trainer import FP8ChaosOptimizer, CUDAGraphTrainer

# 原始混沌优化器 (Henon扰动 + 8-bit/FP32 Adam)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_HERE), 'DeeNeuralNetwork'))
from chaos_optimizer import ChaosGuidedOptimizer

# ═══════════════════════════════════════════
# Part 4: 初始化 & 状态
# ═══════════════════════════════════════════

def init_cuda():
    """初始化CUDA Driver API (加载cuda.dll, 建立context)"""
    _ptx_init()
    sync()
    return get_status()

def get_status():
    """获取CUDA内核状态报告"""
    ptx_dir = _os.path.join(_HERE)
    ptx_files = [f for f in _os.listdir(ptx_dir) if f.endswith('.ptx')]
    
    info = {
        'ptx_kernels': ptx_files,
        'ptx_dir': ptx_dir,
        'triton_available': _has_triton_kernels,
        'optimizers': {
            'fp8_tensor_core': FP8ChaosOptimizer is not None,
            'native_cuda_4bit': NativeCUDAOptimizer is not None,
            'chaos_guided': ChaosGuidedOptimizer is not None,
            'triton_fallback': ChaosTritonOptimizer is not None,
        },
        'cuda_graph': CUDAGraphTrainer is not None,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'vram': f'{torch.cuda.mem_get_info()[1]/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A',
    }
    return info

def print_status():
    """打印内核状态"""
    s = get_status()
    print("=" * 55)
    print("  CUDA Kernels API Status")
    print("=" * 55)
    print(f"  Device: {s['device']} ({s['vram']})")
    print(f"  PTX kernels: {len(s['ptx_kernels'])} compiled")
    for f in s['ptx_kernels']:
        size = _os.path.getsize(_os.path.join(s['ptx_dir'], f))
        print(f"    - {f} ({size}B)")
    print(f"  Triton fallback: {'✓' if s['triton_available'] else '✗'}")
    print(f"  CUDA Graph: {'✓' if s['cuda_graph'] else '✗'}")
    print("=" * 55)

# ═══════════════════════════════════════════
# Part 5: 快捷工厂函数
# ═══════════════════════════════════════════

def create_optimizer(mode='fp8', **kwargs):
    """
    mode: fp8 | 4bit | chaos | triton
    """
    params = kwargs.pop('params')
    
    if mode == 'fp8':
        return FP8ChaosOptimizer(params, **kwargs)
    elif mode == '4bit':
        return NativeCUDAOptimizer(params, **kwargs)
    elif mode == 'chaos':
        return ChaosGuidedOptimizer(params, **kwargs)
    elif mode == 'triton':
        if ChaosTritonOptimizer is None:
            raise RuntimeError("Triton kernels not available")
        return ChaosTritonOptimizer(params, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_trainer(model, vocab, text, mode='fp8', **kwargs):
    """
    mode: fp8_graph | fp8 | 4bit | chaos
    """
    if mode == 'fp8_graph':
        return CUDAGraphTrainer(model, vocab, text, config={
            'lr': kwargs.get('lr', 5e-3),
            'use_cuda_graph': True,
        })
    elif mode in ('fp8', '4bit', 'chaos'):
        opt_cls = { 'fp8': FP8ChaosOptimizer, '4bit': NativeCUDAOptimizer, 
                    'chaos': ChaosGuidedOptimizer }[mode]
        opt = opt_cls(model.parameters(), lr=kwargs.get('lr', 5e-3))
        
        class SimpleTrainer:
            def __init__(self, model, vocab, text, opt):
                self.model = model.train()
                self.vocab = vocab
                self.text = text
                self.opt = opt
                self.device = next(model.parameters()).device
                self.step = 0
                self.loss_history = []
                self.base_lr = kwargs.get('lr', 5e-3)
                self.warmup = kwargs.get('warmup', 100)
                self.vocab_size = kwargs.get('vocab_size', 32000)
                self.seq_len = kwargs.get('seq_len', 128)
            
            def train_step(self):
                import random
                raw = self.text.get_text(200)
                tokens = self.vocab.encode(raw)
                if len(tokens) < self.seq_len + 1:
                    return None
                s = random.randint(0, len(tokens) - self.seq_len - 1)
                chunk = tokens[s:s + self.seq_len + 1]
                x = torch.tensor([chunk[:self.seq_len]], device=self.device)
                y = torch.tensor([chunk[1:self.seq_len + 1]], device=self.device)
                
                if self.step < self.warmup:
                    lr = self.base_lr * max(self.step, 1) / self.warmup
                    for g in self.opt.param_groups:
                        g['lr'] = lr
                
                with torch.amp.autocast('cuda', enabled=True):
                    out = self.model(x)
                    logits = out if not isinstance(out, tuple) else out[0]
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, self.vocab_size), y.reshape(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    return None
                
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.model.zero_grad(set_to_none=True)
                
                self.step += 1
                self.loss_history.append(loss.item())
                return loss.item()
            
            def train(self, steps, log_interval=30):
                import time
                start = time.time()
                last = start
                while self.step < steps:
                    loss = self.train_step()
                    if loss is None: continue
                    now = time.time()
                    if now - last >= log_interval:
                        sps = self.step / max(now - start, 0.01)
                        eta = (steps - self.step) / max(sps, 0.01) / 60
                        avg = sum(self.loss_history[-100:])/min(len(self.loss_history),100)
                        print(f'  [{self.step:>5}] loss={loss:.4f} avg={avg:.4f} '
                              f'{sps:.1f}sps ETA={eta:.0f}m', flush=True)
                        last = now
                print(f'Done: {self.step} steps / {(time.time()-start)/60:.1f}min', flush=True)
                return {'steps': self.step, 'loss': self.loss_history}
            
            def save(self, path):
                import os as _os2
                _os2.makedirs(_os2.dirname(path) or '.', exist_ok=True)
                torch.save({'step': self.step, 'model': self.model.state_dict(),
                           'loss_history': self.loss_history}, path)
        
        return SimpleTrainer(model, vocab, text, opt)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ═══════════════════════════════════════════
# 自动初始化
# ═══════════════════════════════════════════

try:
    _ptx_init()
except Exception as e:
    print(f'[API] PTX init deferred: {e}')

# ═══════════════════════════════════════════
# 公开导出
# ═══════════════════════════════════════════
__all__ = [
    # PTX内核
    'chaos_adam_4bit', 'chaos_adam_fp8', 'henon_perturb', 'dwc_fused',
    'load_ptx', 'get_kernel', 'launch_kernel', 'sync',
    # Triton内核
    'ChaosTritonOptimizer', 'triton_chaos_adam_step',
    # 优化器
    'NativeCUDAOptimizer', 'FP8ChaosOptimizer', 'ChaosGuidedOptimizer',
    # 训练器
    'CUDAGraphTrainer',
    # 工厂
    'create_optimizer', 'create_trainer',
    # 工具
    'init_cuda', 'get_status', 'print_status',
]
