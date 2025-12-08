import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from data_utils.data_dir import DataDir
from methods.contrastive_transformer.model import EventEncoder, info_nce_loss
from methods.contrastive_transformer.data import load_events_df, build_client_sequences, collate_sequences, augment_views

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="Directory with input/target data (split_data output)")
    p.add_argument("--embeddings-dir", type=str, required=True, help="Directory to save embeddings.npy and client_ids.npy")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ffn-mult", type=float, default=2.0, help="Feedforward multiplier relative to d_model for transformer layers")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the transformer encoder")
    p.add_argument("--log-interval", type=int, default=10, help="Number of steps between progress logs")
    p.add_argument("--train-client-frac", type=float, default=1.0, help="Fraction of clients to use for contrastive training")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision training")
    return p


def train_encoder(
    model: EventEncoder,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    temperature: float,
    lr: float,
    log_interval: int,
    use_amp: bool,
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
        total_loss = 0.0
        epoch_start = time.time()
        samples_processed = 0
        for step_idx in range(steps_per_epoch):
            iter_start = time.time()
            start = step_idx * batch_size
            batch_ids = ids[start : start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            v1, v2 = augment_views(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)
                loss = info_nce_loss(z1, z2, temperature=temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_loss = float(loss.detach().cpu())
            total_loss += step_loss
            batch_clients = batch["type_ids"].size(0)
            samples_processed += batch_clients
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps_per_epoch:
                elapsed = time.time() - epoch_start
                avg_step_time = elapsed / (step_idx + 1)
                eta = avg_step_time * (steps_per_epoch - (step_idx + 1))
                samples_per_sec = samples_processed / max(1e-6, elapsed)
                logger.info(
                    "epoch %d/%d | step %d/%d | loss=%.4f | elapsed=%.1fs | eta=%.1fs | %.1f samples/s",
                    ep + 1,
                    epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    step_loss,
                    elapsed,
                    max(0.0, eta),
                    samples_per_sec,
                )
        avg_loss = total_loss / steps_per_epoch
        logger.info(
            "finished epoch %d/%d in %.1fs | avg_loss=%.4f",
            ep + 1,
            epochs,
            time.time() - epoch_start,
            avg_loss,
        )


def generate_embeddings(model: EventEncoder, device: torch.device, client_groups, relevant_client_ids: np.ndarray, batch_size: int, max_seq_len: int, embedding_dim: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    client_ids = relevant_client_ids.astype(np.int64)
    embeddings = np.zeros((client_ids.shape[0], embedding_dim), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, client_ids.shape[0], batch_size):
            batch_ids = client_ids[i : i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            z = model(batch)
            embeddings[i : i + batch_ids.shape[0]] = z.detach().cpu().numpy().astype(np.float16)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventEncoder(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
        ffn_mult=params.ffn_mult,
    )
    model.to(device)
    use_amp = torch.cuda.is_available() and not params.no_amp
    logger.info("Using device=%s (AMP=%s)", device, use_amp)

    logger.info("Training encoder with contrastive InfoNCE...")
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
    params = parser.parse_args()
    main(params)
