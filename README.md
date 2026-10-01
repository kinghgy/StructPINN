# StructPINN

[简体中文](README.zh-CN.md) · [Documentation](docs/en/index.md) · [中文文档](docs/zh-CN/index.md)

StructPINN is a research-oriented, configurable physics-informed neural network solver for static structural mechanics. It currently covers Euler–Bernoulli beams and Kirchhoff–Love rectangular thin plates, with reproducible configurations, independent reference solutions, multi-seed experiments, and CPU/CUDA execution.

> **Research preview:** StructPINN is not a replacement for a verified finite-element package. For a single forward problem, conventional FEM is normally faster and more mature. The useful research targets are physics-constrained learning, inverse problems, parametric surrogates, and method comparison.

## Capabilities

### Beams

- Euler–Bernoulli static bending.
- Uniform, polynomial, sinusoidal, and piecewise-constant distributed loads.
- Multiple interior point loads with analytically embedded shear jumps.
- Constant, linear, or piecewise-constant bending stiffness.
- Fixed, pinned, free, and explicitly prescribed endpoint conditions.
- Deflection, rotation, curvature, bending moment, and shear outputs.
- Independent two-node beam FEM reference.

### Plates

- Kirchhoff–Love static bending of rectangular isotropic plates.
- Uniform, sinusoidal-mode, and Gaussian loads.
- Simply supported, clamped, and straight free-edge conditions.
- Mixed moment formulation and fourth-order displacement formulation.
- Hard constitutive constraints and optional hard trial functions.
- Displacement-derived curvature/moment projection and anchored fine-tuning.
- Navier-series reference for the four-edge simply supported uniform-load case.
- Independent Rayleigh--Ritz reference for rectangular supported/clamped/free-edge
  combinations under uniform, sinusoidal, or Gaussian load.

For free plate edges, the implementation uses the straight-edge Kirchhoff effective shear. Free/free corner concentrated conditions and curved boundaries are not modeled.

### Training and experiments

- Adam followed by optional L-BFGS.
- Uniform, Sobol, and grid sampling.
- Residual-adaptive resampling and gradient-norm loss balancing.
- Dimensionless residual scales, field-branch freezing, compatibility-checked warm starts.
- Mid-Adam checkpoints restoring optimizer, collocation, adaptive weights, best model, and RNG.
- Independent seed sweeps, resumable completed runs, aggregate statistics.
- CPU/CUDA benchmark command and CSV/JSON/PNG result export.

## Verified five-seed result

The included benchmark uses a simply supported plate with `a=1.0`, `b=1.5`, `D=0.1`, `nu=0.3`, and uniform load `q=-2.0`. Seeds are `11, 23, 37, 51, 67`. Values below are mean ± sample standard deviation of relative L2 error after displacement projection and RMS-anchored fine-tuning.

| Field | Relative L2 error |
|---|---:|
| `w` | 0.3342% ± 0.1449% |
| `kx` | 1.2006% ± 0.2281% |
| `ky` | 1.9331% ± 0.2897% |
| `kxy` | 0.8360% ± 0.1283% |
| `Mx` | 1.1489% ± 0.2095% |
| `My` | 1.3934% ± 0.2043% |
| `Mxy` | 0.8360% ± 0.1283% |

On a common diagnostic Sobol grid, the projected fourth-order equilibrium RMS changed from `0.16700 ± 0.02473` to `0.15969 ± 0.02528` (4.38% lower on average). All five seeds improved on this residual. This is still a Pareto improvement, not universal dominance: mean `ky` error increased by 0.23%, and a few seed/field pairs regressed slightly. Mixed-field and projected fourth-order residuals describe different representations and must not be compared as if they were the same metric.

Sanitized data and scope notes are in [`benchmarks/`](benchmarks/README.md).

## Installation

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
```

Activate the environment, then install the package and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For NVIDIA GPUs, install a PyTorch build compatible with the local driver first. `requirements-gpu.txt` records the build used for the published local benchmark; it is not guaranteed to match another machine.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m structpinn.cli check configs/quick_smoke_test.yaml
pytest
```

## Quick start

Run a short CPU beam smoke test:

```bash
python -m structpinn.cli train configs/quick_smoke_test.yaml
```

The smoke configurations use only a few optimizer steps. They verify installation, differentiation, plotting, and checkpoint I/O; they are not accuracy benchmarks.

Run the simply supported plate example:

```bash
python -m structpinn.cli train configs/plate_simply_supported.yaml
```

Run five independent seeds:

```bash
python -m structpinn.cli sweep configs/plate_large_multiseed.yaml \
  --seeds 11 23 37 51 67 \
  --output-root runs/plate_large_multiseed
```

Project a trained plate checkpoint onto displacement-derived fields:

```bash
python -m structpinn.cli predict runs/plate_large_multiseed/seed_11/model.pt \
  --output runs/plate_seed11_projected \
  --plate-field-source displacement
```

Fine-tune matching checkpoints with the verified anchored configuration:

```bash
python -m structpinn.cli finetune-sweep \
  configs/plate_primal_anchored_finetune_gpu.yaml \
  --checkpoint-root runs/plate_large_multiseed \
  --seeds 11 23 37 51 67 \
  --output-root runs/plate_primal_anchored_rms_finetune_gpu
```

The anchored configuration requires existing matching checkpoints; it is not a from-scratch training configuration.

Warm-start a compatible changed condition, or resume an interrupted Adam run:

```bash
python -m structpinn.cli train configs/plate_clamped.yaml \
  --init runs/source/model.pt --output runs/clamped_transfer
python -m structpinn.cli train configs/plate_clamped.yaml \
  --resume runs/clamped_transfer/checkpoints/latest.pt \
  --output runs/clamped_transfer_resumed
```

Warm starting is a continuation heuristic, not a guarantee: large boundary/topology changes can
be slower than training from scratch. See [efficient retraining and recovery](docs/en/workflows.md).

## CLI

| Command | Purpose |
|---|---|
| `check` | Validate a YAML configuration without training. |
| `train` | Train one beam or plate problem. |
| `predict` | Reload a trusted checkpoint and export fields. |
| `sweep` | Train independent seeds and aggregate metrics. |
| `evaluate-sweep` | Evaluate matching checkpoints with one plate field source. |
| `finetune-sweep` | Fine-tune matching seed checkpoints. |
| `benchmark` | Measure the real training loop on requested devices. |

All generated files go under `runs/`, which is intentionally ignored by Git. Publish large checkpoints as versioned release assets only after documenting their configuration and checksum.

## Repository layout

```text
StructPINN/
├── src/structpinn/       package and CLI
├── configs/              reproducible YAML examples and experiments
├── tests/                physics, validation, and workflow tests
├── benchmarks/           sanitized CSV benchmark evidence
├── docs/en/              English documentation
├── docs/zh-CN/           Chinese documentation
├── .github/              CI and contribution templates
└── .run/                 optional shared PyCharm run configurations
```

## Current limitations

- Configuration generality is not one-shot generalization: a changed condition usually requires retraining.
- Navier is limited to the simply supported uniform-load case; other rectangular cases use an
  approximate Ritz reference and still need cross-validation against a mature FEM package.
- Only linear-elastic, static, small-deflection beam and plate theories are implemented.
- Adam checkpoints are recoverable; interrupted L-BFGS currently restarts from the saved
  Adam-complete state rather than restoring an in-progress quasi-Newton history.
- GPU speedup is workload- and hardware-dependent and is modest for the current small FP64 models.

See [validation](docs/en/validation.md), [algorithms](docs/en/algorithms.md), and the [roadmap](docs/en/roadmap.md) before using results in research.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md). StructPINN is available under the [MIT License](LICENSE).
