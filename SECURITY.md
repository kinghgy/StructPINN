# Security / 安全

## English

Report vulnerabilities privately to the repository maintainer rather than opening a public issue. Do not run untrusted configuration files or checkpoints. StructPINN loads checkpoints in PyTorch's restricted `weights_only` mode, but users should still obtain model files from trusted sources. YAML configurations are parsed with `safe_load`.

Generated `runs/`, virtual environments, PDFs, and model binaries are ignored by default because they may contain local paths, personal information, or large files.

## 中文

安全问题应私下报告给仓库维护者，不要直接创建公开 Issue。不要运行来源不明的配置或检查点。StructPINN 使用 PyTorch 受限的 `weights_only` 模式加载检查点，但模型文件仍应来自可信来源；YAML 使用 `safe_load` 解析。

默认忽略 `runs/`、虚拟环境、PDF 和模型二进制，因为它们可能包含本机路径、个人信息或大文件。
