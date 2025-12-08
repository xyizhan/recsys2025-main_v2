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
    load_events_df,
    build_client_sequences,
    collate_sequences,
    augment_views,
)
from methods.contrastive_transformer.model import info_nce_loss
from methods.contrastive_transformer_vae.model import ContrastiveVAEModule


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EVENT_TYPE_IDS = torch.tensor([TYPE_TO_ID[e] for e in EVENT_TYPES], dtype=torch.long)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--embeddings-dir", type=str, required=True)
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ffn-mult", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fusion-weight", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--train-client-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--contrastive-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--recon-weight", type=float, default=1.0)
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


def train_encoder(
    model: ContrastiveVAEModule,
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
    kl_weight: float,
    recon_weight: float,
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

    model.train()
    for ep in range(epochs):
        np.random.shuffle(ids)
        epoch_start = time.time()
        total_loss = 0.0
        total_ctr = 0.0
        total_recon = 0.0
        total_kl = 0.0
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
                recon1 = F.mse_loss(out1["recon"], recon_target)
                recon2 = F.mse_loss(out2["recon"], recon_target)
                recon_loss = 0.5 * (recon1 + recon2)
                kl_loss = 0.5 * (
                    ContrastiveVAEModule.kl_divergence(out1["mu"], out1["logvar"]) +
                    ContrastiveVAEModule.kl_divergence(out2["mu"], out2["logvar"])
                )
                loss = (
                    contrastive_weight * ctr_loss
                    + recon_weight * recon_loss
                    + kl_weight * kl_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_clients = batch["type_ids"].size(0)
            samples_processed += batch_clients
            total_loss += float(loss.detach().cpu())
            total_ctr += float(ctr_loss.detach().cpu())
            total_recon += float(recon_loss.detach().cpu())
            total_kl += float(kl_loss.detach().cpu())

            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps_per_epoch:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - (step_idx + 1))
                logger.info(
                    (
                        "epoch %d/%d | step %d/%d | loss=%.4f | ctr=%.4f | recon=%.4f | kl=%.4f | eta=%.1fs"
                    ),
                    ep + 1,
                    epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    float(loss.detach().cpu()),
                    float(ctr_loss.detach().cpu()),
                    float(recon_loss.detach().cpu()),
                    float(kl_loss.detach().cpu()),
                    max(0.0, eta),
                )

        logger.info(
            "finished epoch %d/%d in %.1fs | loss=%.4f | ctr=%.4f | recon=%.4f | kl=%.4f",
            ep + 1,
            epochs,
            time.time() - epoch_start,
            total_loss / steps_per_epoch,
            total_ctr / steps_per_epoch,
            total_recon / steps_per_epoch,
            total_kl / steps_per_epoch,
        )


def generate_embeddings(
    model: ContrastiveVAEModule,
    device: torch.device,
    client_groups,
    relevant_client_ids: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
):
    model.eval()
    client_ids = relevant_client_ids.astype(np.int64)
    embeddings = np.zeros((client_ids.shape[0], embedding_dim), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, client_ids.shape[0], batch_size):
            batch_ids = client_ids[i : i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, sample_latent=False)
            embeddings[i : i + batch_ids.shape[0]] = out["embed"].detach().cpu().numpy().astype(np.float16)
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

    model = ContrastiveVAEModule(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        latent_dim=params.latent_dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
        ffn_mult=params.ffn_mult,
        fusion_weight=params.fusion_weight,
        recon_dim=len(EVENT_TYPES),
    )
    model.to(device)
    use_amp = device.type == "cuda" and not params.no_amp
    logger.info("Using device=%s (AMP=%s)", device, use_amp)
    if device.type == "cuda":
        dev_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(dev_index))

    logger.info("Training contrastive + VAE encoder...")
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
        kl_weight=params.kl_weight,
        recon_weight=params.recon_weight,
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
    )

    logger.info("Saving embeddings and client_ids to %s", str(embeddings_dir))
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
