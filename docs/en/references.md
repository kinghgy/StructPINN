# References

- M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, 378, 686–707, 2019. [DOI: 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- T. Führer and N. Heuer, “A decomposition result for Kirchhoff plate bending problems and a new discretization approach,” 2017. [arXiv:1703.07962](https://arxiv.org/abs/1703.07962)
- PyTorch documentation, [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) and [deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).

These references support the general PINN framework, Kirchhoff mixed/free-boundary formulation, and reproducibility caveats. They do not validate StructPINN's numerical results; those are supported by the repository tests and benchmark CSV files.
