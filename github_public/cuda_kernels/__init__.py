# CUDA内核 — Triton手写CUDA内核包
# 导入方式:
#   import importlib.util, os
#   path = os.path.join(ROOT, 'CUDA内核', 'chaos_triton_fused.py')
#   spec = importlib.util.spec_from_file_location('ctf', path)
#   mod = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(mod)
#   ChaosTritonOptimizer = mod.ChaosTritonOptimizer
