# Algorithms and equations

## Beam mixed formulation

The network represents deflection `u`, rotation `theta`, curvature `kappa`, and shear `Q`; moment is `M = EI*kappa`. With the project's sign convention,

```text
du/dx - theta = 0
dtheta/dx + kappa = 0
dM/dx - Q = 0
dQ/dx + q = 0
```

Only first spatial derivatives are required. An interior point load `P` at `xp` is embedded as `-P H(x-xp)` in the shear field, enforcing `Q(xp+) - Q(xp-) = -P`. Collocation points exclude a small neighborhood of each discontinuity.

## Plate mixed formulation

Engineering twisting curvature is used consistently:

```text
kx  = -d2w/dx2
ky  = -d2w/dy2
kxy = -2 d2w/dxdy
```

For an isotropic Kirchhoff–Love plate,

```text
Mx  = D (kx + nu ky)
My  = D (ky + nu kx)
Mxy = D (1-nu) kxy / 2
```

and equilibrium is

```text
d2Mx/dx2 + 2 d2Mxy/dxdy + d2My/dy2 + q = 0.
```

`constitutive_enforcement: hard` predicts moment fields and derives curvatures analytically, eliminating a redundant constitutive loss. For a four-edge simply supported plate, trial functions also impose `w=0`, `Mx=0` on the vertical edges, and `My=0` on the horizontal edges.

For a straight free edge, the implemented natural conditions are normal moment and Kirchhoff effective shear. This follows the variational free-boundary condition `∂t Mnt + Div(M)·n = 0`; on a Cartesian straight edge it gives the expressions below ([Führer & Heuer](https://arxiv.org/abs/1703.07962)):

```text
x edge: Mx = 0, Vx = dMx/dx + 2 dMxy/dy = 0
y edge: My = 0, Vy = dMy/dy + 2 dMxy/dx = 0
```

Concentrated corner conditions at intersections of free edges are not included.

## Rayleigh--Ritz plate reference

The independent NumPy reference minimizes Kirchhoff bending potential energy in a tensor-product shifted-Legendre trial space. Edge envelope powers embed `w=0` for simply supported edges and both `w=0` and normal slope zero for clamped edges. Gaussian quadrature assembles the stiffness and load arrays, and natural free-edge conditions follow from stationarity. A condition-number guard rejects rigid or numerically singular systems. This is an approximate weak-form reference, not an exact solution.

## Displacement projection

`output.plate_field_source: displacement` differentiates the displacement branch twice, derives `kx/ky/kxy`, and applies the constitutive law to recover moments. This makes geometry and constitutive relations exact for the reported fields. Its equilibrium diagnostic requires fourth derivatives, so it is reported separately from mixed-field equilibrium.

## RMS-anchored fine-tuning

With `training_formulation: displacement`, the displacement branch can be fine-tuned against fourth-order equilibrium. Direct fine-tuning may reduce residual while degrading field accuracy. `training.reference_anchor` caches the starting checkpoint's displacement-derived fields and penalizes relative changes normalized by each teacher field's RMS. Anchoring requires fixed collocation points and an initialization checkpoint.

## Sampling and loss balancing

Residual terms are normalized before squaring using characteristic load, shear, moment, curvature, rotation, and displacement scales. Residual-adaptive resampling draws from a candidate pool using a residual-based probability with a nonzero offset. Gradient-norm balancing periodically adjusts loss weights with exponential smoothing and bounded weights.

These algorithms are optional. Ablations must keep seeds, points, architecture, optimizer budget, dtype, and validation grid fixed.
