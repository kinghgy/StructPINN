# 配置参考

配置使用 YAML，并与 `src/structpinn/config.py` 中的默认值合并。

## 必需问题数据

梁需要正的 `geometry.length`、`material.EI` 或正刚度分布、分布/集中荷载及左右端边界。

板需要正的 `geometry.a/b`、弯曲刚度 `material.D`、满足 `-1 < nu < 0.5` 的泊松比、荷载和四条边界。程序直接使用 `D`，不会由厚度和弹性模量自动推导。

## 主要训练字段

```yaml
training:
  seed: 42
  diagnostic_seed: 1729
  dtype: float64
  device: auto
  adam_epochs: 5000
  learning_rate: 0.001
  lbfgs_max_iter: 1000
  collocation_points: 256
  boundary_points: 64
  sampler: sobol
  checkpointing:
    enabled: true
    every: 500
    keep_last: 2
```

只有 `torch.cuda.is_available()` 返回 `True` 后才应使用 `device: cuda`。高阶导数经过验证的默认精度是 FP64。

`diagnostic_seed` 单独控制残差报告使用的 Sobol 网格。比较模型时应保持不变；它不控制训练初始化。

## 输出与验证

`output.directory` 控制生成目录。`prediction_points` 对梁表示一维点数，对板表示每个坐标方向的网格点数。`plate_field_source` 可设为 `mixed` 或 `displacement`。

`validation.enabled` 控制参考解比较。`plate_reference` 可设为 `auto`、`navier`、`ritz` 或 `none`；`auto` 优先使用 Navier，否则使用 Ritz。`plate_ritz_order`、`plate_ritz_quadrature`、`plate_convergence_check` 和 `plate_convergence_delta` 控制 Ritz 与自收敛检查。

训练前运行 `python -m structpinn.cli check CONFIG.yaml`。
