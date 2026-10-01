# Validation and interpretation

## Beam reference

Beam predictions are compared with an independent two-node Euler–Bernoulli finite-element implementation. The test suite also checks:

- the analytic midspan deflection of a simply supported uniform beam;
- the analytic tip deflection, rotation, and shear of a cantilever with an end force;
- exact hard endpoint constraints;
- the embedded shear jump at an interior point load.

The variable-stiffness FEM uses elementwise numerical approximation and is a convergence reference, not a closed-form solution.

## Plate reference

The Navier reference is available only when all four edges are simply supported and the load is uniform. It returns displacement, curvature, moments, and shear-resultant components from a truncated double sine series.

Other rectangular cases use an independent NumPy Rayleigh--Ritz energy solver for supported, clamped, and straight free edges with uniform, sinusoidal, or Gaussian loads. A regression test compares Ritz with Navier on their common case. Every reference run can also compare its configured truncation/order with a lower order. This is useful evidence, but not a proof: PINN and Ritz share the same Kirchhoff assumptions, and a general independent plate FEM benchmark is still missing.

## Metrics

- `relative_l2`: global relative L2 error against the reference.
- `rmse`: root mean square absolute error.
- `max_abs`: largest pointwise absolute error on the output grid.
- `normalized_residuals`: RMS and maximum of dimensionless PDE/boundary residuals.
- `projected_normalized_residuals`: fourth-order equilibrium of displacement-derived plate fields.
- `validation_reference.self_convergence`: change between high- and lower-resolution reference solutions.

Weighted training loss is an optimization objective, not an accuracy metric. Report multiple seeds and the sample standard deviation. For point loads, exclude the discontinuity itself from ordinary shear-error statistics and inspect the jump separately.

## Reproducibility boundary

Completed seed resumption now requires the same fully resolved configuration. Residual reporting uses an independent fixed `diagnostic_seed`, so models are compared on common Sobol grids rather than the random state left by training. Floating-point kernels and CUDA libraries may still create small platform-dependent differences; PyTorch explicitly does not guarantee bitwise-identical results across releases, platforms, or CPU/GPU execution ([official numerical-accuracy note](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)).
