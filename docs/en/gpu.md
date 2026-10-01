# GPU notes

The local benchmark was measured on an NVIDIA GeForce RTX 3050 Ti Laptop GPU with the recorded PyTorch CUDA build. It is evidence for that machine, not a portable speed guarantee.

For 300 Adam epochs with 1024 collocation points:

| dtype | CPU mean | CUDA mean | CPU/CUDA ratio |
|---|---:|---:|---:|
| float64 | 41.11 s | 32.46 s | 1.27× |
| float32 | 24.57 s | 32.29 s | 0.76× |

The model is small and uses many higher-order automatic-differentiation operations. Kernel-launch overhead and weak consumer-GPU FP64 throughput limit acceleration. FP32 was faster on CPU in this test and was not adopted for the verified plate results.

Use the official PyTorch installation selector for another machine. A separately installed CUDA Toolkit is normally unnecessary for PyTorch wheels, but the NVIDIA driver must support the selected build.
