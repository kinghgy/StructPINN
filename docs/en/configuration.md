# Configuration reference

Configurations are YAML files merged with defaults in `src/structpinn/config.py`.

## Required problem data

For beams, provide positive `geometry.length`, `material.EI` or a positive profile, distributed/point loads, and left/right boundary conditions.

For plates, provide positive `geometry.a/b`, bending stiffness `material.D`, `-1 < nu < 0.5`, a load, and four edge conditions. `D` is supplied directly; thickness and Young's modulus are not inferred.

## Important training keys

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

Use `device: cuda` only after `torch.cuda.is_available()` returns `True`. FP64 is the verified default for higher derivatives.

`diagnostic_seed` controls the independent Sobol grids used for residual reporting. Keep it fixed when comparing models; it does not control training initialization.

## Output and validation

`output.directory` controls generated files. `prediction_points` is a 1D count for beams and the per-axis grid count for plates. Set `plate_field_source` to `mixed` or `displacement`.

`validation.enabled` controls reference comparison. `plate_reference` is `auto`, `navier`, `ritz`, or `none`; `auto` prefers Navier and otherwise uses Ritz. `plate_ritz_order`, `plate_ritz_quadrature`, `plate_convergence_check`, and `plate_convergence_delta` control the Ritz and self-convergence study.

Run `python -m structpinn.cli check CONFIG.yaml` before training.
