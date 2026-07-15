from cuda_kernels.API import *
# 一行创建优化器
opt = create_optimizer('fp8', params=model.parameters(), lr=5e-3)   # 创建fp8优化器这里是个示例，可以改为我们自己的优化器，支持FP8/FP16/FP32/FP64/INT8/INT16/INT32/INT64等类型,默认是FP8,推荐优化器是FP8/权重是FP32和FP16/FP64混合训练
# 一行创建训练器  
trainer = create_trainer(model, vocab, text, mode='fp8_graph', lr=5e-3)
trainer.train(5000)
# 直接调用内核
chaos_adam_fp8(param, grad, m_fp8, v_fp8, hx, hy, lr=1e-3)
henon_perturb(grad, hx, hy, lam=0.1)
dwc_fused(x, w1, w2, w3)
# 查看状态
print_status()