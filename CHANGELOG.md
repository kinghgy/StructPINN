# Changelog / 更新日志

All notable changes are documented here. This project follows semantic versioning after the first stable release.

所有重要变更记录在此。首个稳定版本发布后将遵循语义化版本规则。

## 0.1.0 — 2026-10-01

### English

- Refactored the thesis prototype into a configurable Python package and CLI.
- Added Euler–Bernoulli beam and Kirchhoff–Love rectangular plate problems.
- Added FEM/Navier validation, adaptive sampling, gradient loss balancing, seed sweeps, device benchmarks, displacement projection, and anchored fine-tuning.
- Added CPU/CUDA configurations, automated tests, bilingual documentation, and sanitized benchmark data.
- Corrected the effective Kirchhoff shear condition on straight free plate edges.
- Added compatibility-checked warm starts for nearby changed conditions.
- Added atomic Adam training-state checkpoints with optimizer, collocation, adaptive-weight, best-model, history, and RNG restoration.
- Added an independent Rayleigh--Ritz rectangular-plate reference and reference-order self-convergence reporting.
- Added deterministic interrupted/resumed training regression tests and bilingual operating guidance.

### 中文

- 将毕设原型重构为可配置 Python 包和命令行工具。
- 增加 Euler–Bernoulli 梁与 Kirchhoff–Love 矩形薄板问题。
- 增加 FEM/Navier 校核、自适应采样、梯度损失配权、多种子实验、设备基准、位移投影及锚定微调。
- 增加 CPU/CUDA 配置、自动测试、双语文档和去标识化基准数据。
- 修正薄板直自由边的 Kirchhoff 有效剪力条件。
- 增加相近变化工况的兼容性检查与迁移初始化。
- 增加原子写入的 Adam 训练状态检查点，恢复优化器、配点、动态权重、最佳模型、历史和随机数状态。
- 增加独立矩形板 Rayleigh--Ritz 参考解及参考阶次自收敛报告。
- 增加确定性中断续训回归测试和双语操作说明。
