#!/usr/bin/env python3
import ctypes
import ctypes.util
import os
import torch
from typing import Dict, Tuple

# ═══════ CUDA Driver API 常量 ═══════
CUDA_SUCCESS = 0
CU_CTX_SCHED_AUTO = 0

# ═══════ 加载 cuda.dll ═══════
_cuda = None

def _get_cuda():
    global _cuda
    if _cuda is not None:
        return _cuda
    
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\cuda.dll",
        r"C:\Windows\System32\nvcuda.dll",
        "nvcuda.dll",
    ]
    
    for path in cuda_paths:
        try:
            _cuda = ctypes.WinDLL(path)
            break
        except OSError:
            continue
    
    if _cuda is None:
        raise RuntimeError("Cannot find cuda.dll. Is CUDA driver installed?")
    # cuInit
    _cuda.cuInit.argtypes = [ctypes.c_uint]
    _cuda.cuInit.restype = ctypes.c_int
    # cuDeviceGet
    _cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    _cuda.cuDeviceGet.restype = ctypes.c_int
    # cuCtxCreate
    _cuda.cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    _cuda.cuCtxCreate.restype = ctypes.c_int
    # cuCtxSetCurrent
    _cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    _cuda.cuCtxSetCurrent.restype = ctypes.c_int
    # cuModuleLoadData
    _cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    _cuda.cuModuleLoadData.restype = ctypes.c_int
    # cuModuleGetFunction
    _cuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    _cuda.cuModuleGetFunction.restype = ctypes.c_int
    # cuLaunchKernel
    _cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,  # f (kernel function)
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid dim
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block dim
        ctypes.c_uint,                                 # shared mem bytes
        ctypes.c_void_p,                               # stream
        ctypes.POINTER(ctypes.c_void_p),               # kernel params
        ctypes.POINTER(ctypes.c_void_p),               # extra
    ]
    _cuda.cuLaunchKernel.restype = ctypes.c_int
    # cuCtxSynchronize
    _cuda.cuCtxSynchronize.argtypes = []
    _cuda.cuCtxSynchronize.restype = ctypes.c_int
    # cuPointerGetAttribute (for getting device pointer from torch tensor)
    _cuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _cuda.cuPointerGetAttribute.restype = ctypes.c_int
    
    # Init
    ret = _cuda.cuInit(0)
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {ret}")
    
    print('[PTX] CUDA Driver API loaded', flush=True)
    return _cuda
_context = None
_module_cache: Dict[str, ctypes.c_void_p] = {}
_kernel_cache: Dict[str, ctypes.c_void_p] = {}
_device_id = 0
def _ensure_context():
    global _context
    cuda = _get_cuda()
    ctx = ctypes.c_void_p()
    ret = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if ret == CUDA_SUCCESS and ctx.value is not None:
        _context = ctx
        return
    torch.zeros(1, device='cuda')

    ret = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if ret == CUDA_SUCCESS and ctx.value is not None:
        _context = ctx
        print('[PTX] Using PyTorch CUDA context', flush=True)
        return
    
    # 最后手段
    device = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(device), 0)
    ret = cuda.cuCtxCreate(ctypes.byref(ctx), CU_CTX_SCHED_AUTO, device)
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxCreate failed: {ret}")
    _context = ctx
    print(f'[PTX] Created new CUDA context on device {device.value}', flush=True)


def load_ptx(ptx_path: str) -> ctypes.c_void_p:
    """加载 .ptx 文件 → CUDA module"""
    if ptx_path in _module_cache:
        return _module_cache[ptx_path]
    cuda = _get_cuda()
    _ensure_context()
    with open(ptx_path, 'rb') as f:
        ptx_data = f.read()
    module = ctypes.c_void_p()
    ret = cuda.cuModuleLoadData(ctypes.byref(module), ptx_data)
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleLoadData failed for {ptx_path}: {ret}")
    _module_cache[ptx_path] = module
    print(f'[PTX] Loaded: {os.path.basename(ptx_path)}', flush=True)
    return module


def get_kernel(ptx_path: str, kernel_name: str) -> ctypes.c_void_p:
    cache_key = f"{ptx_path}:{kernel_name}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]
    
    cuda = _get_cuda()
    module = load_ptx(ptx_path)
    func = ctypes.c_void_p()
    ret = cuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed for {kernel_name}: {ret}")
    _kernel_cache[cache_key] = func
    return func


def launch_kernel(func, grid, block, scalar_args, tensor_ptrs, shared_mem=0, stream=None):
    cuda = _get_cuda()
    _ensure_context()
    if _context is not None:
        cuda.cuCtxSetCurrent(_context)
    all_args = []
    keep_alive = []  
    
    for ptr in tensor_ptrs:
        p = ctypes.c_void_p(ptr)
        keep_alive.append(p)
        all_args.append(ctypes.cast(ctypes.pointer(p), ctypes.c_void_p))
    
    for val in scalar_args:
        keep_alive.append(val)
        all_args.append(ctypes.cast(ctypes.pointer(val), ctypes.c_void_p))
    
    param_array = (ctypes.c_void_p * len(all_args))(*all_args)
    
    grid_x, grid_y, grid_z = (grid[0], grid[1], grid[2]) if len(grid) >= 3 else (grid[0], 1, 1)
    block_x, block_y, block_z = (block[0], block[1], block[2]) if len(block) >= 3 else (block[0], 1, 1)
    ret = cuda.cuLaunchKernel(
        func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        shared_mem,
        stream,
        ctypes.cast(param_array, ctypes.POINTER(ctypes.c_void_p)),
        None,
    )
    if ret != CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {ret}")
def sync():
    """同步 CUDA 流"""
    _get_cuda().cuCtxSynchronize()
#高层 API: 直接调用已编译的内核
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cuda_kernels')

def _ptx(path):
    return os.path.join(_KERNEL_DIR, path)


def chaos_adam_cuda(param, grad, packed, scales, henon_x, henon_y,
                    lr=5e-3, beta1=0.9, beta2=0.95, eps=1e-8,
                    lam=0.1, step=1, chaos_group_size=1024, block_size=256):
    n = param.numel()
    n_blocks = (n + block_size - 1) // block_size
    func = get_kernel(_ptx('chaos_adam_blockwise.ptx'), 'chaos_adam_fused_kernel')
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    
    tensor_ptrs = [
        param.data_ptr(), grad.data_ptr(), packed.data_ptr(), scales.data_ptr(),
        henon_x.data_ptr(), henon_y.data_ptr(),
    ]
    scalar_args = [
        ctypes.c_int(n), ctypes.c_float(lr), ctypes.c_float(beta1),
        ctypes.c_float(beta2), ctypes.c_float(eps), ctypes.c_float(lam),
        ctypes.c_float(bc1), ctypes.c_float(bc2),
    ]
    
    launch_kernel(func, (n_blocks,), (256,), scalar_args, tensor_ptrs)
    sync()
def henon_perturb_cuda(grad, henon_x, henon_y, lam=0.1, lyapunov_factor=1.0):
    """调用 henon_perturb.ptx"""
    n = grad.numel()
    func = get_kernel(_ptx('henon_perturb.ptx'), 'henon_perturb_kernel')
    tensor_ptrs = [grad.data_ptr(), henon_x.data_ptr(), henon_y.data_ptr()]
    scalar_args = [ctypes.c_int(n), ctypes.c_float(lam), ctypes.c_float(lyapunov_factor)]
    threads = 256
    blocks = (n + threads - 1) // threads
    launch_kernel(func, (blocks,), (threads,), scalar_args, tensor_ptrs)
    sync()
def chaos_adam_fp8_cuda(param, grad, m_fp8, v_fp8, henon_x, henon_y,
                         lr=5e-3, beta1=0.9, beta2=0.95, eps=1e-8,
                         lam=0.1, step=1, chaos_group_size=1024, block_size=256):
    """调用 chaos_adam_fp8.ptx — Tensor Core FP8 Adam状态"""
    n = param.numel()
    n_blocks = (n + block_size - 1) // block_size
    func = get_kernel(_ptx('chaos_adam_fp8.ptx'), 'chaos_adam_fp8_kernel')
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    tensor_ptrs = [
        param.data_ptr(), grad.data_ptr(),
        m_fp8.data_ptr(), v_fp8.data_ptr(),
        henon_x.data_ptr(), henon_y.data_ptr(),
    ]
    scalar_args = [
        ctypes.c_int(n), ctypes.c_float(lr), ctypes.c_float(beta1),
        ctypes.c_float(beta2), ctypes.c_float(eps), ctypes.c_float(lam),
        ctypes.c_float(bc1), ctypes.c_float(bc2),
    ]
    
    launch_kernel(func, (n_blocks,), (256,), scalar_args, tensor_ptrs)
    sync()
def dwc_fused_cuda(x, w1, w2, w3):
    """调用 dwc_fused.ptx"""
    d_model = x.shape[0]
    func = get_kernel(_ptx('dwc_fused.ptx'), 'dwc_fused_forward_kernel')
    
    out = torch.empty(d_model, dtype=torch.float32, device=x.device)
    
    tensor_ptrs = [x.data_ptr(), w1.data_ptr(), w2.data_ptr(), w3.data_ptr(), out.data_ptr()]
    scalar_args = [ctypes.c_int(d_model)]
    
    threads = min(256, d_model)
    blocks = (d_model + threads - 1) // threads
    launch_kernel(func, (blocks,), (threads,), scalar_args, tensor_ptrs)
    sync()
    return out
def init():
    _get_cuda()
    _ensure_context()
try:
    init()
except Exception as e:
    print(f'[PTX] Init warning: {e} (will retry on first use)', flush=True)
