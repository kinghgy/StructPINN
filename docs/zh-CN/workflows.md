# 高效重训与恢复

## 工况变化

任意改变几何、材料、荷载或边界后，无法合理保证 PINN 仍能快速收敛。因此 StructPINN
实现的是可测量的延拓流程，而不是宣称一次训练即可泛化：

1. 使用 `train TARGET.yaml --init SOURCE/model.pt` 从相近已解工况启动；
2. 一次只改变一类参数，较大变化优先拆成多个小步；
3. 保持网络布局不变，并与从零训练比较耗时和误差；
4. 若迁移训练停滞或验证退化，则回退到随机初始化。

加载器会拒绝问题类型、网络布局、梁集中力特征数量或板本构输出布局不兼容的检查点。
几何、材料、荷载数值和边界类型可以改变，因为网络坐标域已经归一化。兼容只表示张量可复用，
不等于一定产生正迁移。

若需要在较宽参数范围内反复查询，逐工况热启动不是最终方案。真正的参数化 PINN 或算子学习器
必须把变化参数作为输入，并在未参与训练的参数组合上检验。

## 参考解

`validation.plate_reference: auto` 在满足假设时选用独立 Navier 级数，否则使用 NumPy
Rayleigh--Ritz 能量求解器。Ritz 试函数嵌入齐次本质边界：简支边乘一次因子、固支边乘二次因子、
自由边不乘因子；弯矩和有效剪力自然边界通过弱形式体现。

高阶结果会与较低阶结果比较，差异写入 `metrics.json` 的
`validation_reference.self_convergence`。这只是离散敏感性检查，不能单独证明正确。较强的验证结论至少需要：

- 参考解阶次和积分点加密收敛；
- 在独立网格检查 PDE、边界、几何和本构残差；
- 在共同适用工况下与 Navier 交叉比较；
- 多随机种子统计；
- 对重要应用，再与成熟板有限元程序及网格收敛结果比较。

完全自由的静力板因存在刚体模态而被拒绝。自由边角点力和曲线边界仍不在模型范围内。

## 中断恢复

开启检查点后，程序以原子替换方式写入 `RUN/checkpoints/latest.pt`。文件包含模型、Adam
优化器状态、配点、动态损失权重、历史最佳模型、早停计数、训练历史、累计耗时及
Python/NumPy/Torch 随机数状态。恢复命令为：

```powershell
python -m structpinn.cli train CONFIG.yaml `
  --resume RUN/checkpoints/latest.pt --output RUN_RESUMED
```

恢复时可增大 `training.adam_epochs`，也可改变设备、日志、检查点、输出和验证设置；物理问题、
网络结构、会影响优化过程的训练参数及随机种子必须一致。周期设置如下：

```yaml
training:
  checkpointing:
    enabled: true
    every: 500
    keep_last: 2
```

仓库的确定性 CPU 回归测试验证了中断续训与连续训练参数完全一致，但不同 PyTorch/CUDA 版本或
不同设备之间不承诺逐位相同。正在执行的 L-BFGS 状态目前不保存；中断后应从最近 Adam 检查点
恢复，再重新执行 L-BFGS。
