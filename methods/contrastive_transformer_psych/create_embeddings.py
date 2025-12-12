import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from data_utils.data_dir import DataDir
from methods.contrastive_transformer.data import (
    EVENT_TYPES,
    TYPE_TO_ID,
    augment_views,
    build_client_sequences,
    collate_sequences,
    load_events_df,
)
from methods.contrastive_transformer.model import info_nce_loss
from methods.contrastive_transformer_psych.model import DualPsychVAETransformer, MaslowPsychologicalVAE


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EVENT_TYPE_IDS = torch.tensor([TYPE_TO_ID[e] for e in EVENT_TYPES], dtype=torch.long)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="Directory with split_data output")
    p.add_argument("--embeddings-dir", type=str, required=True, help="Output directory for embeddings.npy and client_ids.npy")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--behavioral-latent-dim", type=int, default=128)
    p.add_argument("--psych-latent-dim", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ffn-mult", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion-behavior", type=float, default=0.5)
    p.add_argument("--fusion-psych", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--train-client-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--behavior-recon-weight", type=float, default=1.0)
    p.add_argument("--behavior-kl-weight", type=float, default=0.1)
    p.add_argument("--psych-recon-weight", type=float, default=0.5)
    p.add_argument("--psych-kl-weight", type=float, default=0.1)
    p.add_argument("--hierarchy-weight", type=float, default=0.05)
    p.add_argument("--enable-mask-predict", action="store_true", help="Enable masked event prediction auxiliary loss")
    p.add_argument("--mask-loss-weight", type=float, default=0.5, help="Weight for masked event prediction loss")
    p.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking event types for auxiliary task")
    p.add_argument("--enable-order-predict", action="store_true", help="Enable sequence order prediction auxiliary loss")
    p.add_argument("--order-loss-weight", type=float, default=0.5, help="Weight for order prediction loss")
    return p


def build_recon_target(type_ids: torch.Tensor) -> torch.Tensor:
    """Return normalized histogram over real event types (excludes MASK/CLS)."""
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
    masked["type_ids"] = torch.where(
        mask_positions,
        torch.full_like(type_ids, mask_token),
        type_ids,
    )
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
            if tensor.dim() == 3:
                ordered[key][flip_mask] = tensor[flip_mask][:, rev_idx, :]
            else:
                ordered[key][flip_mask] = tensor[flip_mask][:, rev_idx]
        labels[flip_mask] = 1
    return ordered, labels


def train_encoder(
    model: DualPsychVAETransformer,
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
    behavior_recon_weight: float,
    behavior_kl_weight: float,
    psych_recon_weight: float,
    psych_kl_weight: float,
    hierarchy_weight: float,
    enable_mask_predict: bool,
    mask_loss_weight: float,
    mask_prob: float,
    enable_order_predict: bool,
    order_loss_weight: float,
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
            "b_recon": 0.0,
            "b_kl": 0.0,
            "p_recon": 0.0,
            "p_kl": 0.0,
            "hierarchy": 0.0,
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
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            recon_target = build_recon_target(batch["type_ids"])
            v1, v2 = augment_views(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                out1 = model(v1, sample_latent=True)
                out2 = model(v2, sample_latent=True)
                ctr_loss = info_nce_loss(out1["embed"], out2["embed"], temperature=temperature)
                behavior_recon = 0.5 * (
                    F.mse_loss(out1["behavior_recon"], recon_target)
                    + F.mse_loss(out2["behavior_recon"], recon_target)
                )
                behavior_kl = 0.5 * (
                    DualPsychVAETransformer.kl_divergence(out1["behavior_mu"], out1["behavior_logvar"])
                    + DualPsychVAETransformer.kl_divergence(out2["behavior_mu"], out2["behavior_logvar"])
                )
                psych_recon = 0.5 * (
                    F.mse_loss(out1["psych_recon"], out1["sequence_hidden"].detach())
                    + F.mse_loss(out2["psych_recon"], out2["sequence_hidden"].detach())
                )
                psych_kl = 0.5 * (
                    MaslowPsychologicalVAE.kl_divergence(out1["psych_mu"], out1["psych_logvar"])
                    + MaslowPsychologicalVAE.kl_divergence(out2["psych_mu"], out2["psych_logvar"])
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
                    else:
                        mask_loss = torch.zeros(1, device=device)
                order_loss = torch.zeros(1, device=device)
                if enable_order_predict:
                    order_batch, order_labels = build_order_prediction_batch(batch)
                    order_logits = model.classify_order(order_batch)
                    order_loss = F.cross_entropy(order_logits, order_labels)
                loss = (
                    contrastive_weight * ctr_loss
                    + behavior_recon_weight * behavior_recon
                    + behavior_kl_weight * behavior_kl
                    + psych_recon_weight * psych_recon
                    + psych_kl_weight * psych_kl
                    + hierarchy_weight * hierarchy
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
            metrics["b_recon"] += float(behavior_recon.detach().cpu())
            metrics["b_kl"] += float(behavior_kl.detach().cpu())
            metrics["p_recon"] += float(psych_recon.detach().cpu())
            metrics["p_kl"] += float(psych_kl.detach().cpu())
            metrics["hierarchy"] += float(hierarchy.detach().cpu())
            if enable_mask_predict:
                metrics["mask"] += float(mask_loss.detach().cpu())
            if enable_order_predict:
                metrics["order"] += float(order_loss.detach().cpu())

            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps_per_epoch:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - (step_idx + 1))
                msg = (
                    "epoch %d/%d | step %d/%d | loss=%.4f | ctr=%.4f | b_recon=%.4f | "
                    "p_recon=%.4f | b_kl=%.4f | p_kl=%.4f | hier=%.4f"
                )
                values = [
                    ep + 1,
                    epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    float(loss.detach().cpu()),
                    float(ctr_loss.detach().cpu()),
                    float(behavior_recon.detach().cpu()),
                    float(psych_recon.detach().cpu()),
                    float(behavior_kl.detach().cpu()),
                    float(psych_kl.detach().cpu()),
                    float(hierarchy.detach().cpu()),
                ]
                if enable_mask_predict:
                    msg += " | mask=%.4f"
                    values.append(float(mask_loss.detach().cpu()))
                if enable_order_predict:
                    msg += " | order=%.4f"
                    values.append(float(order_loss.detach().cpu()))
                msg += " | eta=%.1fs"
                values.append(max(0.0, eta))
                logger.info(msg, *values)
        logger.info(
            "finished epoch %d/%d in %.1fs | loss=%.4f | ctr=%.4f | b_recon=%.4f | p_recon=%.4f | b_kl=%.4f | p_kl=%.4f | hier=%.4f%s%s",
            ep + 1,
            epochs,
            time.time() - epoch_start,
            metrics["loss"] / steps_per_epoch,
            metrics["ctr"] / steps_per_epoch,
            metrics["b_recon"] / steps_per_epoch,
            metrics["p_recon"] / steps_per_epoch,
            metrics["b_kl"] / steps_per_epoch,
            metrics["p_kl"] / steps_per_epoch,
            metrics["hierarchy"] / steps_per_epoch,
            f" | mask={metrics['mask'] / steps_per_epoch:.4f}" if enable_mask_predict else "",
            f" | order={metrics['order'] / steps_per_epoch:.4f}" if enable_order_predict else "",
        )


def generate_embeddings(
    model: DualPsychVAETransformer,
    device: torch.device,
    client_groups,
    relevant_client_ids: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    log_interval: int,
):
    model.eval()
    client_ids = relevant_client_ids.astype(np.int64)
    embeddings = np.zeros((client_ids.shape[0], embedding_dim), dtype=np.float16)
    total = client_ids.shape[0]
    steps = max(1, math.ceil(total / batch_size))
    log_interval = max(1, log_interval)
    start_time = time.time()
    with torch.no_grad():
        for step_idx, i in enumerate(range(0, client_ids.shape[0], batch_size)):
            batch_ids = client_ids[i : i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, sample_latent=False)
            embeddings[i : i + batch_ids.shape[0]] = out["embed"].detach().cpu().numpy().astype(np.float16)
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps:
                processed = min(i + batch_ids.shape[0], total)
                elapsed = time.time() - start_time
                rate = processed / max(1e-6, elapsed)
                logger.info(
                    "embedding gen %d/%d | %d/%d samples | %.1f samples/s",
                    step_idx + 1,
                    steps,
                    processed,
                    total,
                    rate,
                )
    return client_ids, embeddings


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

    device_arg = params.device.lower()
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({params.device}) but torch.cuda.is_available() is False")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = DualPsychVAETransformer(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        behavioral_latent_dim=params.behavioral_latent_dim,
        psych_latent_dim=params.psych_latent_dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
        ffn_mult=params.ffn_mult,
        fusion_behavior=params.fusion_behavior,
        fusion_psych=params.fusion_psych,
        recon_dim=len(EVENT_TYPES),
    )
    model.to(device)
    use_amp = device.type == "cuda" and not params.no_amp
    logger.info("Using device=%s (AMP=%s)", device, use_amp)
    if device.type == "cuda":
        dev_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(dev_index))

    logger.info("Training dual-VAE psychological encoder...")
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
        behavior_recon_weight=params.behavior_recon_weight,
        behavior_kl_weight=params.behavior_kl_weight,
        psych_recon_weight=params.psych_recon_weight,
        psych_kl_weight=params.psych_kl_weight,
        hierarchy_weight=params.hierarchy_weight,
        enable_mask_predict=params.enable_mask_predict,
        mask_loss_weight=params.mask_loss_weight,
        mask_prob=params.mask_prob,
        enable_order_predict=params.enable_order_predict,
        order_loss_weight=params.order_loss_weight,
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
    )

    logger.info("Saving embeddings and client_ids to %s", str(embeddings_dir))
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
