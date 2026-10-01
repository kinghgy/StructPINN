# 参考资料

- M. Raissi、P. Perdikaris、G. E. Karniadakis，“Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations”，*Journal of Computational Physics*，378，686–707，2019。[DOI: 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- T. Führer、N. Heuer，“A decomposition result for Kirchhoff plate bending problems and a new discretization approach”，2017。[arXiv:1703.07962](https://arxiv.org/abs/1703.07962)
- PyTorch 官方文档：[数值精度](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)与[确定性算法](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)。

这些资料用于支持 PINN 总体框架、Kirchhoff 混合/自由边界形式和可复现性限制，不能替代对 StructPINN 数值结果的验证；本项目结果由仓库测试和基准 CSV 支撑。
