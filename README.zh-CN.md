# StructPINN

[English](README.md) · [中文文档](docs/zh-CN/index.md) · [English documentation](docs/en/index.md)

StructPINN 是面向科研、可配置的静力结构力学物理信息神经网络求解器。目前覆盖 Euler–Bernoulli 梁和 Kirchhoff–Love 矩形薄板，并提供可复现配置、独立参考解、多随机种子实验及 CPU/CUDA 运行能力。

> **研究预览版：** StructPINN 不能替代经过充分验证的有限元软件。对于单次正问题，常规有限元通常更快、更成熟。本项目更适合研究物理约束学习、参数反演、参数化代理模型和算法比较。

## 已实现能力

### 梁

- Euler–Bernoulli 梁静力弯曲。
- 均布、多项式、正弦和分段常值分布荷载。
- 多个内部集中力，并解析嵌入剪力跳跃。
- 恒定、线性或分段常值弯曲刚度。
- 固支、铰支、自由及显式给定端部条件。
- 输出挠度、转角、曲率、弯矩和剪力。
- 独立两节点梁有限元参考解。

### 薄板

- 各向同性矩形 Kirchhoff–Love 薄板静力弯曲。
- 均布、正弦模态和高斯荷载。
- 简支、固支及直线自由边条件。
- 混合弯矩形式和四阶位移形式。
- 硬本构约束和可选硬试函数。
- 位移导出曲率/弯矩投影及教师锚定微调。
- 四边简支均布荷载的 Navier 级数参考解。
- 均布、正弦或高斯荷载下矩形板简支/固支/自由边组合的独立 Rayleigh--Ritz 参考解。

自由板边采用直边 Kirchhoff 有效剪力条件；尚未处理自由边交角处的集中角点条件和曲线边界。

### 训练与实验

- Adam 和可选 L-BFGS 两阶段训练。
- 均匀、Sobol 和网格采样。
- 残差自适应采样和梯度范数损失配权。
- 无量纲残差尺度、字段分支冻结、带兼容性检查的迁移初始化。
- Adam 中途检查点，可恢复优化器、配点、动态权重、最佳模型和随机数状态。
- 独立多种子实验、完整种子续跑和汇总统计。
- CPU/CUDA 基准命令及 CSV/JSON/PNG 导出。

## 已验证五种子结果

随仓库提供的基准工况为四边简支板：`a=1.0`、`b=1.5`、`D=0.1`、`nu=0.3`、均布荷载 `q=-2.0`；随机种子为 `11, 23, 37, 51, 67`。下表为位移投影和 RMS 教师锚定微调后的相对 L2 误差均值 ± 样本标准差。

| 物理量 | 相对 L2 误差 |
|---|---:|
| `w` | 0.3342% ± 0.1449% |
| `kx` | 1.2006% ± 0.2281% |
| `ky` | 1.9331% ± 0.2897% |
| `kxy` | 0.8360% ± 0.1283% |
| `Mx` | 1.1489% ± 0.2095% |
| `My` | 1.3934% ± 0.2043% |
| `Mxy` | 0.8360% ± 0.1283% |

在统一诊断 Sobol 网格上，位移投影的四阶平衡 RMS 从 `0.16700 ± 0.02473` 降至 `0.15969 ± 0.02528`，均值下降 4.38%，五个种子均有改善。这仍属于 Pareto 改善，不是所有指标严格胜出：`ky` 平均误差增加 0.23%，少数种子/字段也有轻微退化。混合场残差和位移投影四阶残差属于不同表示，不能当作同一指标直接比较。

去标识化数据及适用范围见 [`benchmarks/`](benchmarks/README.md)。

## 安装

支持 Python 3.10–3.12。

```powershell
python -m venv .venv
```

激活环境后安装项目与开发依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

使用 NVIDIA GPU 时，应先安装与本机驱动兼容的 PyTorch。`requirements-gpu.txt` 只记录本次本机实验使用的版本，不保证适用于其他机器。

检查环境：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m structpinn.cli check configs/quick_smoke_test.yaml
pytest
```

## 快速开始

运行短程 CPU 梁检查：

```powershell
python -m structpinn.cli train configs/quick_smoke_test.yaml
```

快速配置只有少量优化步，只用于检查安装、自动微分、绘图和检查点读写，不是精度基准。

运行四边简支板：

```powershell
python -m structpinn.cli train configs/plate_simply_supported.yaml
```

运行五个独立种子：

```powershell
python -m structpinn.cli sweep configs/plate_large_multiseed.yaml `
  --seeds 11 23 37 51 67 `
  --output-root runs/plate_large_multiseed
```

将板检查点投影为位移导出场：

```powershell
python -m structpinn.cli predict runs/plate_large_multiseed/seed_11/model.pt `
  --output runs/plate_seed11_projected `
  --plate-field-source displacement
```

使用已验证的锚定配置微调对应检查点：

```powershell
python -m structpinn.cli finetune-sweep `
  configs/plate_primal_anchored_finetune_gpu.yaml `
  --checkpoint-root runs/plate_large_multiseed `
  --seeds 11 23 37 51 67 `
  --output-root runs/plate_primal_anchored_rms_finetune_gpu
```

锚定配置必须从已有且种子对应的检查点启动，不能用于从零训练。

对相近新工况进行迁移初始化，或恢复中断的 Adam 训练：

```powershell
python -m structpinn.cli train configs/plate_clamped.yaml `
  --init runs/source/model.pt --output runs/clamped_transfer
python -m structpinn.cli train configs/plate_clamped.yaml `
  --resume runs/clamped_transfer/checkpoints/latest.pt `
  --output runs/clamped_transfer_resumed
```

迁移初始化只是延拓启发式方法，并不保证提速；边界或拓扑变化过大时可能比从零训练更慢。详见[高效重训与恢复](docs/zh-CN/workflows.md)。

## 命令行功能

| 命令 | 用途 |
|---|---|
| `check` | 只检查 YAML 配置，不训练。 |
| `train` | 训练一个梁或板问题。 |
| `predict` | 重新加载可信检查点并导出场。 |
| `sweep` | 训练多个独立种子并汇总指标。 |
| `evaluate-sweep` | 使用统一板场来源评估对应检查点。 |
| `finetune-sweep` | 微调对应种子的检查点。 |
| `benchmark` | 在指定设备上测量真实训练循环。 |

生成文件统一写入 `runs/`，该目录默认不提交 Git。大型模型应作为带版本的 GitHub Release 资产发布，并同时给出配置和校验值。

## 目录结构

```text
StructPINN/
├── src/structpinn/       求解器包和命令行
├── configs/              可复现 YAML 示例与实验配置
├── tests/                物理、参考解及工作流测试
├── benchmarks/           去标识化 CSV 基准证据
├── docs/en/              英文文档
├── docs/zh-CN/           中文文档
├── .github/              CI 和贡献模板
└── .run/                 可选 PyCharm 共享运行配置
```

## 当前限制

- 配置层面的泛用性不等于单模型泛化；工况变化后通常仍需重新训练。
- Navier 只覆盖四边简支均布荷载；其他矩形工况使用近似 Ritz 参考，仍需成熟有限元软件交叉验证。
- 只实现线弹性、静力、小挠度梁板理论。
- Adam 检查点可恢复；L-BFGS 若在阶段内部中断，目前会从 Adam 完成态重启，不能恢复正在进行的拟牛顿历史。
- GPU 加速取决于硬件和任务；当前小型 FP64 网络的提升有限。

研究使用前请阅读[验证说明](docs/zh-CN/validation.md)、[算法说明](docs/zh-CN/algorithms.md)和[路线图](docs/zh-CN/roadmap.md)。

## 贡献与许可证

请阅读 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。项目采用 [MIT License](LICENSE)。
