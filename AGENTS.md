# Repository Guidelines

## Project Structure & Module Organization
- `methods/` hosts embedding generators. Notable entries include `contrastive_transformer`, `contrastive_transformer_vae`, and the new `contrastive_transformer_psych` dual-VAE variant; each exposes a `create_embeddings.py` CLI plus model definitions.
- `data_utils/` contains loaders (`DataDir`, `load_with_properties`) that every embedding script reuses for reading `input/` and `target/` chunks.
- `training_pipeline/` and `validator/` mirror the official challenge flow for downstream evaluation and entry validation.
- `baseline/` stores reference notebooks/scripts. Logs from long runs should land in `logs/` (created by the provided shell scripts).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` sets up the environment (PyTorch + Lightning, validator deps).
- `python -m methods.contrastive_transformer.create_embeddings --data-dir <split_data> --embeddings-dir <out>` runs the baseline contrastive Transformer.
- `python -m methods.contrastive_transformer_vae.create_embeddings ...` enables the VAE head; `run_contrastive_vae_embedding.sh` wraps common defaults.
- `python -m methods.contrastive_transformer_psych.create_embeddings ...` launches the dual-vae psychological encoder.
- `python -m validator.run --data-dir <split_data> --embeddings-dir <out>` checks submission format before packaging.

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation, type hints where practical. Favor explicit module paths (`methods.contrastive_transformer.data`) to keep CLIs importable.
- Hash-bucketing logic for categorical IDs lives inside each model class; keep naming consistent (`emb_*`, `*_latent_dim`).
- Logging via `logging` module; avoid print except for quick scripts.

## Testing Guidelines
- Unit tests are lightweight; rely on scripted sanity checks (short local runs with `--epochs 1 --batch-size 64`). When adding new data transforms, craft minimal synthetic batches to cover padding/masking.
- Follow numpy output contract: save `client_ids.npy` (`int64`) and `embeddings.npy` (`float16`), align ordering, and verify via `validator`.

## Commit & Pull Request Guidelines
- Commit messages generally follow `type(scope): summary`, e.g., `feat: 添加对比学习VAE嵌入生成脚本`. Use Mandarin descriptions when that matches existing history.
- Keep PRs focused: describe motivation, list major code paths touched (`methods/...`, `training_pipeline/...`), and mention how you validated (commands run, datasets). Attach logs or screenshots for long runs, and link any tracker issue IDs when available.
