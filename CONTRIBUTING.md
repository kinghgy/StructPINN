# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

Contributions are welcome when they preserve reproducibility and distinguish numerical evidence from claims.

1. Create a focused branch and keep generated `runs/`, checkpoints, PDFs, local paths, and personal data out of commits.
2. Install with `python -m pip install -e ".[dev]"`.
3. Add or update tests. New equations, loads, and boundary conditions require a manufactured solution, analytic result, or independent numerical reference.
4. Run `ruff check src tests` and `pytest`.
5. Update both English and Chinese documentation for user-visible changes.
6. In the pull request, state the configuration, seeds, dtype, device, acceptance metric, and known regressions.

Do not describe a configuration-driven solver as a one-shot generalizing model. Do not claim improvement from a single seed or a training loss alone.
