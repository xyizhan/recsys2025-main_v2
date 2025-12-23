import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from data_utils.data_dir import DataDir
from methods.contrastive_transformer.data import (
    load_events_df,
    build_client_sequences,
    TYPE_TO_ID,
)
from methods.contrastive_tgn.data import collate_tgn_sequences
from methods.contrastive_tgn.model import TGNEncoder, sampled_ranking_loss


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--item-bucket-count", type=int, default=100_000)
    parser.add_argument("--delta-hidden-dim", type=int, default=128)
    parser.add_argument("--delta-scale", type=float, default=3600.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train-client-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def train_encoder(
    model: TGNEncoder,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    log_interval: int,
    num_negatives: int,
    use_amp: bool,
    amp_device_type: str,
):
    client_ids = list(client_groups.keys())
    if len(client_ids) == 0:
        logger.warning("No clients available for training. Skipping fit phase.")
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = max(1, math.ceil(len(client_ids) / batch_size))
    log_interval = max(1, log_interval)

    for epoch in range(epochs):
        np.random.shuffle(client_ids)
        epoch_loss = 0.0
        start_time = time.time()
        for step_idx in range(steps_per_epoch):
            start = step_idx * batch_size
            batch_ids = client_ids[start : start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_tgn_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                output = model(batch)
                loss = sampled_ranking_loss(
                    user_embeddings=output.user_embeddings,
                    pos_item_embeddings=output.pos_item_embeddings,
                    lengths=output.lengths,
                    item_embedding=model.item_embedding,
                    item_bucket_count=model.item_bucket_count,
                    num_negatives=num_negatives,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps_per_epoch:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / (step_idx + 1)
                logger.info(
                    "epoch %d/%d | step %d/%d | loss=%.4f | elapsed=%.1fs",
                    epoch + 1,
                    epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    avg_loss,
                    elapsed,
                )
        logger.info(
            "finished epoch %d/%d | avg_loss=%.4f | epoch_time=%.1fs",
            epoch + 1,
            epochs,
            epoch_loss / steps_per_epoch,
            time.time() - start_time,
        )


def generate_embeddings(
    model: TGNEncoder,
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
    steps = max(1, math.ceil(client_ids.shape[0] / batch_size))
    log_interval = max(1, log_interval)
    start_time = time.time()
    with torch.no_grad():
        for step_idx, i in enumerate(range(0, client_ids.shape[0], batch_size)):
            batch_ids = client_ids[i : i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_tgn_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = model.generate_embeddings(batch)
            embeddings[i : i + batch_ids.shape[0]] = (
                emb.detach().cpu().numpy().astype(np.float16)
            )
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == steps:
                processed = min(i + batch_ids.shape[0], client_ids.shape[0])
                elapsed = time.time() - start_time
                rate = processed / max(1e-6, elapsed)
                logger.info(
                    "embedding gen %d/%d | %d/%d samples | %.1f samples/s",
                    step_idx + 1,
                    steps,
                    processed,
                    client_ids.shape[0],
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
    client_groups = build_client_sequences(
        dfs=dfs,
        relevant_client_ids=relevant_client_ids,
    )

    train_groups = client_groups
    if 0 < params.train_client_frac < 1.0 and len(client_groups) > 0:
        rng = np.random.default_rng(params.seed)
        keep = max(1, int(len(client_groups) * params.train_client_frac))
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
            raise RuntimeError(
                f"CUDA device requested ({params.device}) but torch.cuda.is_available() is False"
            )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    type_vocab_size = max(TYPE_TO_ID.values()) + 1
    model = TGNEncoder(
        type_vocab_size=type_vocab_size,
        item_bucket_count=params.item_bucket_count,
        embed_dim=params.embedding_dim,
        delta_hidden_dim=params.delta_hidden_dim,
        dropout=params.dropout,
        delta_scale=params.delta_scale,
    )
    model.to(device)
    use_amp = device.type == "cuda" and not params.no_amp
    logger.info("Using device=%s (AMP=%s)", device, use_amp)
    if device.type == "cuda":
        dev_index = device.index if device.index is not None else torch.cuda.current_device()
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(dev_index))

    logger.info("Training TGN encoder...")
    train_encoder(
        model=model,
        device=device,
        client_groups=train_groups,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        epochs=params.epochs,
        lr=params.lr,
        log_interval=params.log_interval,
        num_negatives=params.num_negatives,
        use_amp=use_amp,
        amp_device_type="cuda" if device.type == "cuda" else "cpu",
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

