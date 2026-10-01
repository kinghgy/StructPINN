# Configurations / 配置

## English

Files in this directory are supported examples or recommended reproducible workflows. Start with `quick_smoke_test.yaml`, `beam_uniform.yaml`, or `plate_simply_supported.yaml`.

`plate_primal_anchored_finetune_gpu.yaml` is a fine-tuning configuration and requires matching existing checkpoints. It cannot be run from scratch with the `train` command.

Historical ablations are under `experiments/`. They are retained for reproducibility, not as recommended settings.

## 中文

本目录文件是受支持的示例或推荐复现流程。建议从 `quick_smoke_test.yaml`、`beam_uniform.yaml` 或 `plate_simply_supported.yaml` 开始。

`plate_primal_anchored_finetune_gpu.yaml` 是微调配置，必须具有随机种子对应的已有检查点，不能通过 `train` 从零运行。

历史消融配置位于 `experiments/`，只为复现保留，不代表推荐参数。
