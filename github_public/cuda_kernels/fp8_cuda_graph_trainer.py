#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import os, sys, time, random
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cuda_kernels'))
from ptx_loader import chaos_adam_fp8_cuda, henon_perturb_cuda


class FP8ChaosOptimizer(torch.optim.Optimizer):

    
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
                nh = triton.cdiv(n, self.chaos_group_size)
                self._states[pid] = {
                    'm_fp8': torch.zeros(n, dtype=torch.uint8, device=dev),
                    'v_fp8': torch.zeros(n, dtype=torch.uint8, device=dev),
                    'hx': torch.rand(nh, device=dev) * 0.2 - 0.1,
                    'hy': torch.rand(nh, device=dev) * 0.2 - 0.1,
                }
                total_bytes += n * 2 + nh * 8
        
        self._init_done = True
        params_M = sum(s['m_fp8'].numel() for s in self._states.values()) / 1e6
        print(f'[FP8-Adam] {params_M:.1f}M参数, '
              f'状态: {total_bytes/1e9:.2f}GB (FP8, 4x vs FP32)', flush=True)
    
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
                
                chaos_adam_fp8_cuda(
                    p.data.view(-1),
                    p.grad.data.view(-1),
                    s['m_fp8'],
                    s['v_fp8'],
                    s['hx'],
                    s['hy'],
                    lr=lr, beta1=b1, beta2=b2, eps=eps,
                    lam=lam, step=self.step_count,
                    chaos_group_size=self.chaos_group_size,
                    block_size=self.block_size,
                )

class CUDAGraphTrainer:
    def __init__(self, model, vocab, text, config=None):
        cfg = config or {}
        self.model = model.train()
        self.vocab = vocab
        self.text = text
        self.device = next(model.parameters()).device
        self.step = 0
        self.loss_history = []
        
        # FP8优化器
        self.opt = FP8ChaosOptimizer(
            model.parameters(),
            lr=cfg.get('lr', 5e-3),
            lambda_init=cfg.get('lambda_init', 0.1),
            lambda_decay=cfg.get('lambda_decay', 0.999)
        )
        
        self.warmup = cfg.get('warmup', 100)
        self.base_lr = cfg.get('lr', 5e-3)
        self.vocab_size = cfg.get('vocab_size', 32000)
        self.seq_len = cfg.get('seq_len', 128)
        
        # CUDA Graph
        self._graph = None
        self._graph_pool = None
        self._static_x = None
        self._static_y = None
        self._use_graph = cfg.get('use_cuda_graph', True)
        
        params = sum(p.numel() for p in model.parameters())
        vram = torch.cuda.memory_allocated() / 1e9
        print(f'[CUDAGraph] {params/1e6:.1f}M params, '
              f'FP8+Graph, VRAM={vram:.1f}GB', flush=True)
    
    def _capture_graph(self):
        if not self._use_graph or self._graph is not None:
            return
        
        print('[CUDAGraph] Capturing training graph...', flush=True)
        self.model.zero_grad(set_to_none=True)
        
        raw = self.text.get_text(200)
        tokens = self.vocab.encode(raw)
        s = random.randint(0, len(tokens) - self.seq_len - 1)
        chunk = tokens[s:s + self.seq_len + 1]        
        self._static_x = torch.zeros(1, self.seq_len, dtype=torch.long, device=self.device)
        self._static_y = torch.zeros(1, self.seq_len, dtype=torch.long, device=self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.amp.autocast('cuda', enabled=True):
                out = self.model(self._static_x)
                logits = out if not isinstance(out, tuple) else out[0]
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    self._static_y.reshape(-1)
                )
            loss.backward()
        
        self._graph = g
        print('[CUDAGraph] Graph captured ✓', flush=True)
    
    def train_step(self):
        raw = self.text.get_text(200)
        tokens = self.vocab.encode(raw)
        if len(tokens) < self.seq_len + 1:
            return None
        
        s = random.randint(0, len(tokens) - self.seq_len - 1)
        chunk = tokens[s:s + self.seq_len + 1]
        x = torch.tensor([chunk[:self.seq_len]], device=self.device)
        y = torch.tensor([chunk[1:self.seq_len + 1]], device=self.device)
        
        if step_counter := getattr(self, 'step', 0) < self.warmup:
            lr = self.base_lr * max(step_counter, 1) / self.warmup
            for g in self.opt.param_groups:
                g['lr'] = lr
        
        if self._use_graph and self._graph is not None:
            self._static_x.copy_(x)
            self._static_y.copy_(y)
            self._graph.replay()
        else:
            self.model.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=True):
                out = self.model(x)
                logits = out if not isinstance(out, tuple) else out[0]
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    y.reshape(-1)
                )
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                out = self.model(x)
                logits = out if not isinstance(out, tuple) else out[0]
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    y.reshape(-1)
                )
        
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.model.zero_grad(set_to_none=True)
        
        self.step += 1
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def train(self, steps, log_interval=30):
        self._capture_graph()
        
        start_time = time.time()
        last_log = start_time
        
        print(f'[Train] Starting {steps} steps (FP8 + CUDA Graph)...', flush=True)
        while self.step < steps:
            loss = self.train_step()
            if loss is None:
                continue
            vram = torch.cuda.memory_allocated() / 1e9
            if vram > 11.0:
                torch.cuda.empty_cache()
            now = time.time()
            if now - last_log >= log_interval or self.step == 1:
                elapsed = now - start_time
                sps = self.step / max(elapsed, 0.01)
                eta = (steps - self.step) / max(sps, 0.01) / 60
                avg100 = sum(self.loss_history[-100:]) / min(len(self.loss_history), 100)
                print(f'  [{self.step:>5}] loss={loss:.4f} avg100={avg100:.4f} '
                      f'{sps:.1f}sps ETA={eta:.0f}m', flush=True)
                last_log = now
        
        elapsed = time.time() - start_time
        print(f'\nDone: {self.step} steps / {elapsed/60:.1f}min', flush=True)
        return {'steps': self.step, 'loss_history': self.loss_history}
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'step': self.step,
            'model': self.model.state_dict(),
            'loss_history': self.loss_history,
        }, path)
        print(f'[Save] {path}', flush=True)
