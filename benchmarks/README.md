# Benchmarks / 基准数据

## English

The CSV files are sanitized copies of locally measured experiments. They contain no checkpoints, personal information, or absolute paths.

- `device_float64.csv`: three CPU/CUDA repetitions, 300 Adam epochs, 1024 collocation points, FP64.
- `device_float32.csv`: the matching FP32 experiment.
- `plate_five_seed_runs.csv`: optimized plate metrics for seeds 11, 23, 37, 51, and 67.
- `plate_five_seed_projected_runs.csv`: matching displacement-projection metrics before anchored fine-tuning.
- `plate_five_seed_comparison.csv`: aggregate displacement-projection baseline versus RMS-anchored fine-tuning.

In the raw CSV, `relative_l2` values are fractions (`0.01` means 1%). Standard deviations use the sample definition with denominator `n-1`. Device timing is hardware-specific. The plate comparison applies only to the condition documented in the root README; it does not establish general performance across boundary conditions.

Model weights are intentionally excluded from Git. Reproduce them from the configurations or publish trusted weights as GitHub Release assets with a checksum.

## 中文

这些 CSV 是本机实测实验的去标识化副本，不包含检查点、个人信息或绝对路径。

- `device_float64.csv`：FP64、1024 配点、300 个 Adam 步的三次 CPU/CUDA 重复。
- `device_float32.csv`：对应的 FP32 实验。
- `plate_five_seed_runs.csv`：种子 11、23、37、51、67 的优化板指标。
- `plate_five_seed_projected_runs.csv`：锚定微调前对应的位移投影指标。
- `plate_five_seed_comparison.csv`：位移投影基线与 RMS 锚定微调的汇总对比。

原始 CSV 中 `relative_l2` 使用小数表示（`0.01` 表示 1%）。标准差为分母 `n-1` 的样本标准差。设备耗时只适用于实测硬件。板对比仅适用于根 README 记录的工况，不能证明其他边界工况下的普遍性能。

Git 中不包含模型权重。可以按配置重新训练，或将可信权重作为带校验值的 GitHub Release 资产发布。
