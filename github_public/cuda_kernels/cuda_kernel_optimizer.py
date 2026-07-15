#!/usr/bin/env python3
"""
cuda_kernel_optimizer.py — PyTorch优化器, 使用编译好的PTX CUDA内核
================================================================
直接调用 chaos_adam_blockwise.ptx + henon_perturb.ptx, 绕过Triton.
速度: 原生CUDA, 无JIT编译开销.
"""
import torch
import triton
import os, sys

# 添加父目录以导入 ptx_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptx_loader import chaos_adam_cuda


class NativeCUDAOptimizer(torch.optim.Optimizer):
    """
    原生CUDA优化器 — 编译好的PTX内核, 无Triton依赖
    
    存储: 0.98GB packed(uint8) + 8MB scales(fp16) + Henon state = ~1.0GB
    """
    
    def __init__(self, params, lr=5e-3, betas=(0.9, 0.95), eps=1e-8,
                 lambda_init=0.1, lambda_decay=0.999,
                 chaos_group_size=1024, block_size=256):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                       lambda_init=lambda_init, lambda_decay=lambda_decay)
        super().__init__(params, defaults)
        
        self.step_count = 0
        self.chaos_group_size = chaos_group_size
        self.block_size = block_size
        self._states = {}
        self._init_done = False
    
    def _ensure_init(self):
        if self._init_done:
            return
        
        total_bytes = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad or p.numel() == 0:
                    continue
                
                pid, n, dev = id(p), p.numel(), p.device
                bs = self.block_size
                nb = triton.cdiv(n, bs)
                nh = triton.cdiv(n, self.chaos_group_size)
                
                self._states[pid] = {
                    'packed': torch.full((n,), 0x8F, dtype=torch.uint8, device=dev),
                    'scales': torch.ones(nb, dtype=torch.float16, device=dev),
                    'hx': torch.rand(nh, device=dev) * 0.2 - 0.1,
                    'hy': torch.rand(nh, device=dev) * 0.2 - 0.1,
                }
                total_bytes += n * 1 + nb * 2 + nh * 8
        
        self._init_done = True
        params_M = sum(s['packed'].numel() for s in self._states.values()) / 1e6
        print(f'[NativeCUDA] {params_M:.1f}M参数, '
              f'优化器: {total_bytes/1e9:.2f}GB', flush=True)
    
    @torch.no_grad()
    def step(self, closure=None):
        self.step_count += 1
        self._ensure_init()
        
        for group in self.param_groups:
            lam = group['lambda_init'] * (group['lambda_decay'] ** self.step_count)
            lr, (b1, b2), eps = group['lr'], group['betas'], group['eps']
            
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    continue
                
                s = self._states[id(p)]
                
                chaos_adam_cuda(
                    p.data.view(-1),
                    p.grad.data.view(-1),
                    s['packed'],
                    s['scales'],
                    s['hx'],
                    s['hy'],
                    lr=lr, beta1=b1, beta2=b2, eps=eps,
                    lam=lam, step=self.step_count,
                    chaos_group_size=self.chaos_group_size,
                    block_size=self.block_size,
                )
