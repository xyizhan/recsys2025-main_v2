import argparse
import logging
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from data_utils.data_dir import DataDir
from methods.contrastive_transformer.model import info_nce_loss
from methods.contrastive_transformer_psych.data import (
    EVENT_TYPES,
    TYPE_TO_ID,
    augment_views,
    build_client_sequences,
    collate_sequences,
    load_events_df,
    STATS_DIM,
)
from methods.contrastive_transformer_dualcls.model import (
    DeepInterestVAE,
    DualCLSContrastiveTransformer,
    ShallowInterestVAE,
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EVENT_TYPE_IDS = torch.tensor([TYPE_TO_ID[e] for e in EVENT_TYPES], dtype=torch.long)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--shallow-latent-dim", type=int, default=96)
    parser.add_argument("--deep-latent-dim", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ffn-mult", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--short-window", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train-client-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--shallow-recon-weight", type=float, default=1.0)
    parser.add_argument("--shallow-kl-weight", type=float, default=0.1)
    parser.add_argument("--deep-recon-weight", type=float, default=0.5)
    parser.add_argument("--deep-kl-weight", type=float, default=0.1)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--enable-mask-predict", action="store_true")
    parser.add_argument("--mask-loss-weight", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--enable-order-predict", action="store_true")
    parser.add_argument("--order-loss-weight", type=float, default=0.2)
    parser.add_argument("--stats-dim", type=int, default=STATS_DIM)
    parser.add_argument("--feat-path", type=str, required=True)
    return parser


def build_recon_target(type_ids: torch.Tensor) -> torch.Tensor:
    device = type_ids.device
    type_ids = type_ids.clamp(min=0)
    num_classes = max(TYPE_TO_ID.values()) + 1
    one_hot = F.one_hot(type_ids, num_classes=num_classes).float()
    mask = (type_ids >= 2).float().unsqueeze(-1)
    one_hot = one_hot * mask
    hist = one_hot.sum(dim=1)
    cols = EVENT_TYPE_IDS.to(device)
    hist = hist.index_select(dim=1, index=cols)
    hist_sum = hist.sum(dim=1, keepdim=True)
    hist = torch.where(hist_sum > 0, hist / hist_sum.clamp(min=1e-6), torch.zeros_like(hist))
    return hist


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in batch.items()}


def build_mask_prediction_batch(batch: dict[str, torch.Tensor], mask_prob: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    mask_prob = float(max(0.0, min(1.0, mask_prob)))
    masked = _clone_batch(batch)
    type_ids = masked["type_ids"]
    device = type_ids.device
    mask_labels = torch.full_like(type_ids, fill_value=-100)
    if mask_prob == 0.0:
        return masked, mask_labels
    rand = torch.rand(type_ids.shape, device=device)
    mask_positions = rand < mask_prob
    mask_labels[mask_positions] = type_ids[mask_positions]
    mask_token = TYPE_TO_ID["MASK"]
    masked["type_ids"] = torch.where(mask_positions, torch.full_like(type_ids, mask_token), type_ids)
    return masked, mask_labels


def build_order_prediction_batch(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ordered = _clone_batch(batch)
    type_ids = ordered["type_ids"]
    device = type_ids.device
    B, L = type_ids.shape
    labels = torch.zeros(B, dtype=torch.long, device=device)
    if L <= 1:
        return ordered, labels
    flip_mask = torch.rand(B, device=device) < 0.5
    if flip_mask.any():
        rev_idx = torch.arange(L - 1, -1, -1, device=device)
        for key, tensor in ordered.items():
            if tensor.dim() == 3 and tensor.size(1) == L:
                ordered[key][flip_mask] = tensor[flip_mask][:, rev_idx, :]
            elif tensor.dim() == 2 and tensor.size(1) == L:
                ordered[key][flip_mask] = tensor[flip_mask][:, rev_idx]
        labels[flip_mask] = 1
    return ordered, labels


def train_encoder(
    model: DualCLSContrastiveTransformer,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    temperature: float,
    lr: float,
    log_interval: int,
    use_amp: bool,
    amp_device_type: str,
    contrastive_weight: float,
    shallow_recon_weight: float,
    shallow_kl_weight: float,
    deep_recon_weight: float,
    deep_kl_weight: float,
    structure_weight: float,
    enable_mask_predict: bool,
    mask_loss_weight: float,
    mask_prob: float,
    enable_order_predict: bool,
    order_loss_weight: float,
    stats_feature: dict,
):
    ids = list(client_groups.keys())
    if len(ids) == 0:
        logger.warning("No clients available for training. Skipping fit phase.")
        return
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_interval = max(1, log_interval)
    steps_per_epoch = max(1, math.ceil(len(ids) / batch_size))
    logger.info(
        "Training on %d clients (%d steps/epoch, batch_size=%d)",
        len(ids),
        steps_per_epoch,
        batch_size,
    )
    for ep in range(epochs):
        np.random.shuffle(ids)
        epoch_start = time.time()
        metrics = {
            "loss": 0.0,
            "ctr": 0.0,
            "s_recon": 0.0,
            "s_kl": 0.0,
            "d_recon": 0.0,
            "d_kl": 0.0,
            "struct": 0.0,
        }
        if enable_mask_predict:
            metrics["mask"] = 0.0
        if enable_order_predict:
            metrics["order"] = 0.0
        samples_processed = 0
        for step_idx in range(steps_per_epoch):
            start = step_idx * batch_size
            batch_ids = ids[start : start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            feats = [stats_feature.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len, stats_feat=feats)
            batch = {k: v.to(device) for k, v in batch.items()}
            recon_target = build_recon_target(batch["type_ids"])
            v1, v2 = augment_views(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                out1 = model(v1, sample_latent=True)
                out2 = model(v2, sample_latent=True)
                ctr_loss = info_nce_loss(out1["embed"], out2["embed"], temperature=temperature)
                shallow_recon = 0.5 * (
                    F.mse_loss(out1["shallow_recon"], recon_target)
                    + F.mse_loss(out2["shallow_recon"], recon_target)
                )
                shallow_kl = 0.5 * (
                    ShallowInterestVAE.kl_divergence(out1["shallow_mu"], out1["shallow_logvar"])
                    + ShallowInterestVAE.kl_divergence(out2["shallow_mu"], out2["shallow_logvar"])
                )
                deep_recon = 0.5 * (
                    F.mse_loss(out1["deep_recon"], out1["deep_target"].detach())
                    + F.mse_loss(out2["deep_recon"], out2["deep_target"].detach())
                )
                deep_kl = 0.5 * (
                    DeepInterestVAE.kl_divergence(out1["deep_mu"], out1["deep_logvar"])
                    + DeepInterestVAE.kl_divergence(out2["deep_mu"], out2["deep_logvar"])
                )
                hierarchy = 0.5 * (out1["hierarchy_loss"] + out2["hierarchy_loss"])
                mask_loss = torch.zeros(1, device=device)
                if enable_mask_predict:
                    masked_batch, mask_labels = build_mask_prediction_batch(batch, mask_prob)
                    mask_logits = model.predict_mask_logits(masked_batch)
                    vocab = mask_logits.size(-1)
                    loss_inputs = mask_logits.reshape(-1, vocab)
                    loss_targets = mask_labels.reshape(-1)
                    if (loss_targets != -100).any():
                        mask_loss = F.cross_entropy(loss_inputs, loss_targets, ignore_index=-100)
                order_loss = torch.zeros(1, device=device)
                if enable_order_predict:
                    order_batch, order_labels = build_order_prediction_batch(batch)
                    order_logits = model.classify_order(order_batch)
                    order_loss = F.cross_entropy(order_logits, order_labels)
                loss = (
                    contrastive_weight * ctr_loss
                    + shallow_recon_weight * shallow_recon
                    + shallow_kl_weight * shallow_kl
                    + deep_recon_weight * deep_recon
                    + deep_kl_weight * deep_kl
                    + structure_weight * hierarchy
                )
                if enable_mask_predict:
                    loss = loss + mask_loss_weight * mask_loss
                if enable_order_predict:
                    loss = loss + order_loss_weight * order_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_clients = batch["type_ids"].size(0)
            samples_processed += batch_clients
            metrics["loss"] += float(loss.detach().cpu())
            metrics["ctr"] += float(ctr_loss.detach().cpu())
            metrics["s_recon"] += float(shallow_recon.detach().cpu())
            metrics["s_kl"] += float(shallow_kl.detach().cpu())
            metrics["d_recon"] += float(deep_recon.detach().cpu())
            metrics["d_kl"] += float(deep_kl.detach().cpu())
            metrics["struct"] += float(hierarchy.detach().cpu())
            if enable_mask_predict:
                metrics["mask"] += float(mask_loss.detach().cpu())
            if enable_order_predict:
                metrics["order"] += float(order_loss.detach().cpu())
            if (step_idx + 1) % log_interval == 0:
                avg = {k: v / max(1, step_idx + 1) for k, v in metrics.items()}
                logger.info(
                    "epoch %d/%d step %d/%d loss=%.4f ctr=%.4f shallow=(recon %.4f kl %.4f) deep=(recon %.4f kl %.4f) struct=%.4f",
                    ep + 1,
                    epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    avg["loss"],
                    avg["ctr"],
                    avg["s_recon"],
                    avg["s_kl"],
                    avg["d_recon"],
                    avg["d_kl"],
                    avg["struct"],
                )
        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d finished in %.1fs (%.1f samples/s)",
            ep + 1,
            elapsed,
            samples_processed / max(1e-6, elapsed),
        )


def generate_embeddings(
    model: DualCLSContrastiveTransformer,
    device: torch.device,
    client_groups,
    relevant_client_ids: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    log_interval: int,
    stats_feature: dict,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    client_ids: list[int] = []
    embeddings: list[np.ndarray] = []
    steps = math.ceil(len(relevant_client_ids) / batch_size)
    with torch.no_grad():
        for step_idx in range(steps):
            start = step_idx * batch_size
            batch_ids = relevant_client_ids[start : start + batch_size]
            seqs = [client_groups.get(int(cid)) for cid in batch_ids]
            feats = [stats_feature.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len, stats_feat=feats)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, sample_latent=False)
            emb = out["embed"].detach().cpu().numpy().astype(np.float32)
            client_ids.append(batch_ids)
            embeddings.append(emb)
            if (step_idx + 1) % max(1, log_interval) == 0:
                processed = min(len(relevant_client_ids), (step_idx + 1) * batch_size)
                logger.info("Embedding generation %d/%d (%d/%d clients)", step_idx + 1, steps, processed, len(relevant_client_ids))
    client_ids_arr = np.concatenate(client_ids).astype(np.int64)
    embeddings_arr = np.concatenate(embeddings).astype(np.float16)
    return client_ids_arr, embeddings_arr


def main(params):
    data_dir = DataDir(Path(params.data_dir))
    embeddings_dir = Path(params.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("medium")
    except AttributeError:
        pass
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    logger.info("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    logger.info("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)
    train_groups = client_groups
    if 0 < params.train_client_frac < 1.0 and len(client_groups) > 0:
        rng = np.random.default_rng(params.seed)
        keep = max(1, int(len(client_groups) * params.train_client_frac))
        if keep < len(client_groups):
            selected = set(rng.choice(list(client_groups.keys()), size=keep, replace=False))
            train_groups = {cid: client_groups[cid] for cid in selected}
            logger.info(
                "Subsampled %d/%d clients (%.1f%%) for faster training",
                len(train_groups),
                len(client_groups),
                100 * len(train_groups) / max(1, len(client_groups)),
            )
    with open(Path(params.feat_path), "rb") as f:
        stats_feature = pickle.load(f)
    device_arg = params.device.lower()
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({params.device}) but torch.cuda.is_available() is False")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = DualCLSContrastiveTransformer(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        shallow_latent_dim=params.shallow_latent_dim,
        deep_latent_dim=params.deep_latent_dim,
        recon_dim_shallow=len(EVENT_TYPES),
        recon_dim_deep=params.d_model * 2,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
        ffn_mult=params.ffn_mult,
        short_window=params.short_window,
        stats_dim=params.stats_dim,
    )
    model.to(device)
    use_amp = device.type == "cuda" and not params.no_amp
    logger.info("Using device=%s (AMP=%s)", device, use_amp)
    if device.type == "cuda":
        dev_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(dev_index))
    logger.info("Training dual-CLS encoder...")
    train_encoder(
        model=model,
        device=device,
        client_groups=train_groups,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        epochs=params.epochs,
        temperature=params.temperature,
        lr=params.lr,
        log_interval=params.log_interval,
        use_amp=use_amp,
        amp_device_type="cuda" if device.type == "cuda" else "cpu",
        contrastive_weight=params.contrastive_weight,
        shallow_recon_weight=params.shallow_recon_weight,
        shallow_kl_weight=params.shallow_kl_weight,
        deep_recon_weight=params.deep_recon_weight,
        deep_kl_weight=params.deep_kl_weight,
        structure_weight=params.structure_weight,
        enable_mask_predict=params.enable_mask_predict,
        mask_loss_weight=params.mask_loss_weight,
        mask_prob=params.mask_prob,
        enable_order_predict=params.enable_order_predict,
        order_loss_weight=params.order_loss_weight,
        stats_feature=stats_feature,
    )
    logger.info("Generating embeddings for relevant clients...")
    client_ids, embeddings = generate_embeddings(
        model=model,
        device=device,
        client_groups=client_groups,
        relevant_client_ids=relevant_client_ids,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        embedding_dim=params.embedding_dim,
        log_interval=params.log_interval,
        stats_feature=stats_feature,
    )
    logger.info("Saving embeddings and client_ids to %s", str(embeddings_dir))
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)


if __name__ == "__main__":
    main(get_parser().parse_args())
