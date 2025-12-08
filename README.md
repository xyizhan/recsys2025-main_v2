# Universal Behavioral Modeling Challenge 仓库说明

本仓库复现并扩展了 Synerise RecSys 2025「Universal Behavioral Modeling Challenge」的基线方案。项目目标是根据 `input/` 目录下的行为日志，为 `relevant_clients.npy` 中的 100 万用户生成可复用的行为表征，然后在公开/隐藏的下游任务中验证泛化能力。本文档概述数据组织方式、主要模型族以及运行流程，帮助你快速完成本地实验与提交。

## 目录结构
- `methods/contrastive_transformer/`：原始的对比学习 Transformer，SimCLR-style 投影 + InfoNCE 监督。
- `methods/contrastive_transformer_vae/`：在序列编码后串联单 VAE 头（行为重构 + KL 正则）。
- `methods/contrastive_transformer_psych/`：新增的双 VAE 结构，兼顾行为潜变量与基于马斯洛层次的心理潜变量。
- `data_utils/`：`DataDir`、`load_with_properties` 等数据访问组件。
- `training_pipeline/`：官方下游模型与计算图，可用于自检。
- `validator/`：提交前的格式校验脚本。
- `baseline/`：参考 notebook/脚本，便于复现论文中的基线指标。
- `run_contrastive_vae_embedding.sh`、`run_contrastive_psych_embedding.sh`：封装常用 CLI 启动命令并输出日志到 `logs/`。

## 环境与数据准备
1. Python 3.11，安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 将官方数据拆分为 `input/`（行为日志与 `relevant_clients.npy`）和 `target/`（下游监督信号）；参考原仓库的 `split_data` 说明或 `README_ori.md`。
3. `data_utils/data_dir.py` 默认假设目录结构如下：
   ```
   <data-root>/
     input/
       relevant_clients.npy
       events_<type>.parquet
     target/
       churn.parquet
       ...
   ```

## 模型族与特性
| 模型 | 主要特征 | 损失项 |
| --- | --- | --- |
| `contrastive_transformer` | Transformer Encoder + SimCLR 投影 | InfoNCE |
| `contrastive_transformer_vae` | 共享 Transformer 表征 + 行为 VAE 重构分支 | InfoNCE + 行为重构 MSE + KL |
| `contrastive_transformer_psych` | 行为 VAE + 马斯洛层级心理 VAE 双分支，支持层次依赖门控与多路融合 | InfoNCE + 行为/心理重构 + 双 KL + 层次约束 |

所有模型均复用 `methods/contrastive_transformer/data.py` 中的数据处理、增广与 `collate_sequences`，输入输出接口一致：生成 `client_ids.npy (int64)` 与 `embeddings.npy (float16)`，顺序一一对应。

## 运行示例
以 `../autodl-tmp/recsys_new` 为数据目录、`../autodl-tmp/embeddings_out/...` 为输出目录：

```bash
# 纯对比学习
python -m methods.contrastive_transformer.create_embeddings \
  --data-dir ../autodl-tmp/recsys_new \
  --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_v1 \
  --epochs 1 --batch-size 512 --max-seq-len 256 --device cuda:0

# Transformer + 单VAE（等同 run_contrastive_vae_embedding.sh）
bash run_contrastive_vae_embedding.sh

# Transformer + 双VAE（新增脚本）
bash run_contrastive_psych_embedding.sh
```

常用可调参数：
- `--train-client-frac`：子采样部分用户以提速。
- `--behavioral-latent-dim / --psych-latent-dim`：控制 VAE 潜空间大小。
- `--contrastive-weight / --behavior-*/--psych-*`：平衡多任务损失。
- `--log-interval`：训练日志频率，长任务建议 ≥20。

## 验证与提交
1. 确保输出目录含 `client_ids.npy` 与 `embeddings.npy` 且 dtype 正确，embedding 维度 ≤ 2048。
2. 使用官方校验器：
   ```bash
   python -m validator.run \
     --data-dir ../autodl-tmp/recsys_new \
     --embeddings-dir ../autodl-tmp/embeddings_out/contrastive_transformer_psych_v1
   ```
3. 若需要离线评估，可在 `training_pipeline/` 中运行公开任务训练脚本并比较 AUROC/Novelty/Diversity。

## 贡献说明
- 贡献建议参考 `AGENTS.md`（贡献者指南）。核心原则：保持输入输出契约不变、对齐日志/目录命名、在 PR 描述中附运行命令或日志。
- 新增方法请放置在 `methods/` 下独立子目录，并重用 `data.py` 数据接口以保证 pipeline 兼容。
