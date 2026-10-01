# GPU 说明

本机基准来自 NVIDIA GeForce RTX 3050 Ti Laptop GPU 和仓库记录的 PyTorch CUDA 构建，只能证明该机器上的结果，不能作为其他设备的速度保证。

1024 配点、300 个 Adam 步的结果：

| 精度 | CPU 平均 | CUDA 平均 | CPU/CUDA 比值 |
|---|---:|---:|---:|
| float64 | 41.11 s | 32.46 s | 1.27× |
| float32 | 24.57 s | 32.29 s | 0.76× |

当前网络较小，并包含大量高阶自动微分算子。内核启动开销和消费级 GPU 较弱的 FP64 吞吐限制了加速。该试验中 FP32 在 CPU 上反而更快，因此已验证板结果仍使用 FP64。

其他机器应使用 PyTorch 官方安装选择器。PyTorch wheel 通常不需要单独安装 CUDA Toolkit，但 NVIDIA 驱动必须支持所选构建。
