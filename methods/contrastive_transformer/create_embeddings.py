import argparse
import logging
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
    return p


def train_encoder(model: EventEncoder, device: torch.device, client_groups, batch_size: int, max_seq_len: int, epochs: int, temperature: float, lr: float):
    ids = list(client_groups.keys())
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(ids)
        total_loss = 0.0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            v1, v2 = augment_views(batch)
            z1 = model(v1)
            z2 = model(v2)
            loss = info_nce_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        logger.info(f"epoch={ep+1}/{epochs} train_loss={total_loss/ max(1,len(ids)//batch_size):.4f}")


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

    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    logger.info("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    logger.info("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventEncoder(d_model=params.d_model, embed_dim=params.embedding_dim, n_layers=params.n_layers, n_heads=params.n_heads)
    model.to(device)

    logger.info("Training encoder with contrastive InfoNCE...")
    train_encoder(
        model=model,
        device=device,
        client_groups=client_groups,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        epochs=params.epochs,
        temperature=params.temperature,
        lr=params.lr,
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