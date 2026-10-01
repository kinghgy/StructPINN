# GitHub release checklist / GitHub 发布清单

## Before the first push / 首次推送前

- [x] Source, tests, supported configs, sanitized benchmarks, license, CI, and bilingual documentation are present. / 源码、测试、受支持配置、去标识化基准、许可证、CI 和双语文档齐全。
- [x] PDFs, checkpoints, full runs, caches, local paths, and personal data are excluded. / 已排除 PDF、检查点、完整运行结果、缓存、本机路径和个人信息。
- [x] Tests, lint, checkpoint reload, deterministic diagnostics, and wheel build have passed. / 测试、静态检查、检查点加载、诊断复现和 wheel 构建均已通过。
- [ ] Replace the generic contributor identity only if a public author name is desired. / 只有在希望公开作者身份时才替换通用贡献者名称。
- [x] Create an empty GitHub repository and record its URL. / 创建空 GitHub 仓库并记录地址。

Repository / 仓库：[kinghgy/StructPINN](https://github.com/kinghgy/StructPINN)

## Suggested commands / 建议命令

```bash
git init -b main
git add .
git status
git commit -m "Initial public release of StructPINN"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Review `git status` before committing. Do not force-add ignored PDF, `runs/`, or model files. / 提交前检查 `git status`，不要强制添加被忽略的 PDF、`runs/` 或模型文件。

## Optional model release / 可选模型发布

If trained weights are published, use a GitHub Release rather than normal Git history. Include the exact commit, resolved YAML configuration, seed, dtype, device, metrics, file size, and SHA-256 checksum. / 如需发布训练权重，请使用 GitHub Release，不要写入普通 Git 历史；同时提供准确提交、解析配置、种子、精度、设备、指标、文件大小和 SHA-256。
