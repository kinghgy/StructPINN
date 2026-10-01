# 算法与方程

## 梁混合形式

网络表示挠度 `u`、转角 `theta`、曲率 `kappa` 和剪力 `Q`，弯矩为 `M = EI*kappa`。项目采用以下符号约定：

```text
du/dx - theta = 0
dtheta/dx + kappa = 0
dM/dx - Q = 0
dQ/dx + q = 0
```

因此只需要一阶空间导数。内部集中力 `P` 位于 `xp` 时，在剪力场中嵌入 `-P H(x-xp)`，严格满足 `Q(xp+) - Q(xp-) = -P`。配点会避开不连续位置的小邻域。

## 板混合形式

统一采用工程扭曲曲率：

```text
kx  = -d2w/dx2
ky  = -d2w/dy2
kxy = -2 d2w/dxdy
```

各向同性 Kirchhoff–Love 薄板本构关系为：

```text
Mx  = D (kx + nu ky)
My  = D (ky + nu kx)
Mxy = D (1-nu) kxy / 2
```

平衡方程为：

```text
d2Mx/dx2 + 2 d2Mxy/dxdy + d2My/dy2 + q = 0.
```

设置 `constitutive_enforcement: hard` 时，网络预测弯矩并解析恢复曲率，从而消除冗余本构损失。四边简支时，试函数还会严格满足 `w=0`、竖直边 `Mx=0` 和水平边 `My=0`。

直线自由边采用法向弯矩和 Kirchhoff 有效剪力条件。变分自由边条件为 `∂t Mnt + Div(M)·n = 0`，在笛卡尔直边上得到下式（参见 [Führer 与 Heuer](https://arxiv.org/abs/1703.07962)）：

```text
x 边：Mx = 0，Vx = dMx/dx + 2 dMxy/dy = 0
y 边：My = 0，Vy = dMy/dy + 2 dMxy/dx = 0
```

自由边交点处的集中角点条件尚未实现。

## Rayleigh--Ritz 板参考解

独立 NumPy 参考程序在张量积移位 Legendre 试函数空间内最小化 Kirchhoff 弯曲势能。边界包络的一次因子嵌入简支边 `w=0`，二次因子嵌入固支边 `w=0` 和法向斜率为零。程序用 Gauss 积分组装刚度与荷载数组，自由边自然条件由驻值原理得到，并用条件数检查拒绝刚体或数值奇异系统。这是近似弱形式参考解，不是精确解。

## 位移投影

`output.plate_field_source: displacement` 对位移分支求二阶导数，得到 `kx/ky/kxy`，再由本构关系恢复弯矩。报告场会严格满足几何和本构关系。其平衡诊断需要四阶导数，因此必须与混合弯矩场平衡残差分开报告。

## RMS 教师锚定微调

设置 `training_formulation: displacement` 后，可以用四阶平衡方程微调位移分支。直接微调可能降低残差却破坏场精度。`training.reference_anchor` 会缓存初始检查点的位移导出场，并按各教师场自身 RMS 归一化，惩罚相对漂移。锚定微调要求固定配点和初始检查点。

## 采样与损失配权

残差在平方前使用荷载、剪力、弯矩、曲率、转角和位移特征尺度无量纲化。残差自适应采样从候选点池中按残差概率重采样，并保留非零偏置。梯度范数配权定期调整损失权重，使用指数平滑和上下界限制。

这些算法均可关闭。消融实验必须保持随机种子、配点、网络、优化预算、精度和验证网格一致。
