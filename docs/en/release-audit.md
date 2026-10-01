# Release audit — 2026-10-01

## Scope

This audit covers the public `StructPINN` tree, not the private thesis PDF or the complete local experiment archive.

## Corrections made before release

1. Corrected straight free-plate-edge effective shear from `Mx,x + Mxy,y` to `Mx,x + 2 Mxy,y`, with the analogous y-edge expression. Added a manufactured-field unit test. The published five-seed plate benchmark uses only simply supported edges and is unaffected.
2. Residual diagnostics now use a separate fixed `diagnostic_seed`. Re-evaluating all ten before/after checkpoints on common Sobol grids left every field-error statistic unchanged and corrected the mean fourth-order residual improvement from 5.33% to 4.38%.
3. Completed-run resumption now compares the full resolved configuration, not only the seed and file presence.
4. Checkpoints are loaded using PyTorch `weights_only=True`.
5. Historical failed/diagnostic configurations were separated from recommended configurations and labeled explicitly.
6. Raw runs, PDFs, model binaries, local absolute paths, caches, and personal data were excluded.
7. Added a NumPy Rayleigh--Ritz cross-check for rectangular plate cases outside the Navier scope, with order-refinement diagnostics.
8. Added atomic Adam checkpoints and exact same-environment CPU resume regression coverage. In-progress L-BFGS history remains outside the recovery scope.

## Verification performed

- 39 automated tests passed on Python 3.11 with the local CUDA-enabled PyTorch environment.
- `ruff check .` and `pip check` passed.
- All YAML configurations, shared PyCharm XML files, local Markdown links, citation metadata, and benchmark evidence were checked by tests.
- Benchmark summary values were recomputed from both five-seed raw CSV tables.
- An end-to-end beam smoke training completed and saved/reloaded a checkpoint.
- A trusted historical plate checkpoint loaded in restricted mode; its seed-11 displacement relative L2 error was `0.1746237%` and its common-grid projected equilibrium RMS was `0.1533744`.
- A pure-Python wheel built successfully as `structpinn-0.1.0-py3-none-any.whl`.
- Adam checkpoint save/resume completed on both CPU and CUDA; the CPU split-run parameters matched uninterrupted training exactly.

## Claims intentionally not made

- General plate accuracy is not established by a mature FEM benchmark outside the Navier case; Ritz is an independent approximation and convergence cross-check, not proof.
- The solver is not a one-shot model for unseen conditions.
- The smoke configuration is not an accuracy benchmark.
- GPU speedup is not guaranteed on other hardware.
- Free/free corner concentrated conditions are not implemented.
