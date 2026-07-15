#!/usr/bin/env python3
# chaos_triton_fused.py自洽Block-wise混沌Adam融合内核 (Triton++)
import torch
import triton
import triton.language as tl
@triton.jit
def _chaos_adam_blockwise_kernel(
    param_ptr, grad_ptr, packed_ptr, scale_ptr,
    henon_x_ptr, henon_y_ptr,
    n_elements: tl.constexpr,
    lr: tl.constexpr, beta1: tl.constexpr, beta2: tl.constexpr,
    eps: tl.constexpr, lam: tl.constexpr,
    bc1: tl.constexpr, bc2: tl.constexpr,
    chaos_group_size: tl.constexpr, QBLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    start = block_id * QBLOCK
    offsets = start + tl.arange(0, QBLOCK)
    mask = offsets < n_elements
    # 加载
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    packed = tl.load(packed_ptr + offsets, mask=mask, other=0x8F)
    old_scale = tl.load(scale_ptr + block_id).to(tl.float32)
    # 解量化
    m_q = (packed >> 4).to(tl.float32)
    v_q = (packed & 0x0F).to(tl.float32)
    m = (m_q - 8.0) * old_scale
    v = (v_q - 8.0) * old_scale * 0.1
    v = tl.maximum(v, 1e-8)
    # Adam
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad * grad
    # Henon
    hen_id = offsets // chaos_group_size
    hx = tl.load(henon_x_ptr + hen_id, mask=mask, other=0.0)
    hy = tl.load(henon_y_ptr + hen_id, mask=mask, other=0.0)
    hx_new = 1.0 - 1.4 * hx * hx + hy
    hy_new = 0.3 * hx
    tl.store(henon_x_ptr + hen_id, hx_new, mask=mask)
    tl.store(henon_y_ptr + hen_id, hy_new, mask=mask)
    chaos_pert = hx_new * lam * 0.01
    # 参数更新
    denom = tl.sqrt(v / bc2) + eps
    update = (lr / bc1) * m / denom + chaos_pert
    tl.store(param_ptr + offsets, param - update, mask=mask)
    # 重新量化 (自洽: 用同block的新m计算scale)
    m_abs_max = tl.max(tl.abs(m))
    new_scale = m_abs_max / 7.0 + 1e-8
    m_q_new = tl.clamp(m / new_scale + 8.0, 0.0, 15.0)
    v_q_new = tl.clamp(v / (new_scale * 0.1) + 8.0, 0.0, 15.0)
    packed_new = (m_q_new.to(tl.uint8) << 4) | (v_q_new.to(tl.uint8) & 0x0F)
    tl.store(packed_ptr + offsets, packed_new, mask=mask)
    tl.store(scale_ptr + block_id, new_scale.to(tl.float16))
def triton_chaos_adam_step(param, grad, packed, scales, hx, hy,
                            lr=5e-3, beta1=0.9, beta2=0.95, eps=1e-8,
                            lam=0.1, step=1, chaos_group_size=1024,
                            block_size=256):
    n = param.numel()
    n_blocks = triton.cdiv(n, block_size)
    assert n_blocks == scales.numel()
    
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    _chaos_adam_blockwise_kernel[(n_blocks,)](
        param, grad, packed, scales, hx, hy,
        n, lr, beta1, beta2, eps, lam, bc1, bc2,
        chaos_group_size, QBLOCK=block_size,
    )
class ChaosTritonOptimizer(torch.optim.Optimizer):
#混沌Triton优化器: 自洽block-wise 4-bit Adam + Henon扰动
    
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
                    'scales': torch.full((nb,), 10.0, dtype=torch.float16, device=dev),
                    'hx': torch.rand(nh, device=dev) * 0.2 - 0.1,
                    'hy': torch.rand(nh, device=dev) * 0.2 - 0.1,
                }
                total_bytes += n * 1 + nb * 2 + nh * 8
        self._init_done = True
        params_M = sum(s['packed'].numel() for s in self._states.values()) / 1e6
        print(f'[ChaosTriton] {params_M:.1f}M参数, '
              f'优化器: {total_bytes/1e9:.2f}GB (vs 8bitAdam {params_M*2/1e6:.2f}GB)', flush=True)
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
                triton_chaos_adam_step(
                    p.data.view(-1), p.grad.data.view(-1),
                    s['packed'], s['scales'], s['hx'], s['hy'],
                    lr=lr, beta1=b1, beta2=b2, eps=eps,
                    lam=lam, step=self.step_count,
                    chaos_group_size=self.chaos_group_size,
                    block_size=self.block_size,
                )
