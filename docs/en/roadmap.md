# Open problems and roadmap

## Unresolved issues

1. Mixed moment fields achieve low second-order equilibrium residuals, while displacement-derived fields achieve better geometry/constitutive consistency. A single representation does not yet dominate both objectives.
2. General plate cases have an independent Ritz cross-check but still lack a mature FEM reference.
3. Seed variance remains material; single-seed conclusions are unsafe.
4. Warm starts reduce some nearby-condition costs, but broad parameter generalization still requires a parametric model.
5. Adaptive sampling does not explicitly reserve quotas for corners and boundary layers.
6. Adam state is recoverable; in-progress L-BFGS history is not.
7. The scope is linear-elastic, static, and small-deflection.

## Priorities

### P0 — credibility and recovery

- Independent plate FEM benchmarks for clamped, free, mixed, and localized-load cases.
- L-BFGS-state recovery and multi-objective checkpoint selection.
- Stratified interior/boundary/corner sampling and unified Pareto reporting.

### P1 — condition generalization

- Parametric PINNs with geometry, stiffness, Poisson ratio, and load parameters as inputs.
- Controlled comparison with DeepONet/FNO under equal data and compute budgets.
- Load identification, material inversion, and noise sensitivity.

### P2 — structural theory

- Timoshenko beams, Reissner–Mindlin plates, anisotropic laminates.
- Modal/transient response, buckling, and geometric/material nonlinearity.
- Uncertainty estimates and ensembles.

### P3 — application layer

- A small GUI/API only after the solver and general validation are mature.
- CSV/VTK export, section plots, contour comparison, and packaged result archives.
