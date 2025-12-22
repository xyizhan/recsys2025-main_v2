# transformer_mlm_baseline.py
import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle

from data_utils.data_dir import DataDir
from methods.contrastive_transformer_psych.data import (
    EVENT_TYPES,
    TYPE_TO_ID,
    build_client_sequences,
    collate_sequences,
    load_events_df,
    STATS_DIM
)
from methods.contrastive_transformer.model import PositionalEncoding

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
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--train-client-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking event types for MLM")
    return p

def build_mask_prediction_batch(batch: dict, mask_prob: float = 0.15):
    masked = {k: v.clone() for k, v in batch.items()}
    type_ids = masked["type_ids"]
    device = type_ids.device
    
    mask_labels = torch.full_like(type_ids, fill_value=-100)
    
    if mask_prob > 0:
        rand = torch.rand(type_ids.shape, device=device)
        mask_positions = rand < mask_prob
        mask_labels[mask_positions] = type_ids[mask_positions]
        
        mask_token = TYPE_TO_ID.get("MASK", 0)
        masked["type_ids"] = torch.where(
            mask_positions,
            torch.full_like(type_ids, mask_token),
            type_ids,
        )
    
    return masked, mask_labels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        length = x.size(1)
        return x + self.pe[:length].unsqueeze(0)

# Transformer MLM模型 类似BERT4Rec
class TransformerMLM(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0
    ):
        super().__init__()
        self.d_model = d_model
        
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        
        # [CLS] token和位置编码
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)
        
        # Transformer编码器
        dim_ff = max(d_model, int(d_model * ffn_mult))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, type_buckets)
        )
        
        # 最终嵌入投影
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, embed_dim)
        )
        
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_events(self, batch):
        x = self.emb_type(self._hash(batch["type_ids"], self.emb_type.num_embeddings))
        
        if "sku_ids" in batch:
            x = x + self.emb_sku(self._hash(batch["sku_ids"], self.emb_sku.num_embeddings))
        if "cat_ids" in batch:
            x = x + self.emb_cat(self._hash(batch["cat_ids"], self.emb_cat.num_embeddings))
        if "price_ids" in batch:
            x = x + self.emb_price(self._hash(batch["price_ids"], self.emb_price.num_embeddings))
        if "url_ids" in batch:
            x = x + self.emb_url(self._hash(batch["url_ids"], self.emb_url.num_embeddings))
        if "query_vec" in batch:
            x = x + self.query_proj(batch["query_vec"])
        
        x = self.ln(self.gelu(x))
        return x

    def forward(self, batch, mlm_labels=None):
        x = self.encode_events(batch)
        B, L, _ = x.shape

        cls_token = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        cls_output = x[:, 0, :]
        sequence_output = x[:, 1:, :]
        mlm_logits = self.mlm_head(sequence_output)
        
        embed = self.proj(cls_output)
        embed = F.normalize(embed, dim=-1)
        
        return {
            "embed": embed,
            "mlm_logits": mlm_logits,
            "cls_output": cls_output
        }

def train_mlm(
    model: TransformerMLM,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    log_interval: int,
    use_amp: bool,
    amp_device_type: str,
    mask_prob: float,
):
    ids = list(client_groups.keys())
    if len(ids) == 0:
        logger.warning("No clients available for training")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log_interval = max(1, log_interval)
    steps_per_epoch = max(1, math.ceil(len(ids) / batch_size))
    
    logger.info(f"Training on {len(ids)} clients ({steps_per_epoch} steps/epoch)")

    for ep in range(epochs):
        np.random.shuffle(ids)
        epoch_start = time.time()
        total_loss = 0
        total_mlm_loss = 0
        
        for step_idx in range(steps_per_epoch):
            start = step_idx * batch_size
            batch_ids = ids[start: start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            masked_batch, mlm_labels = build_mask_prediction_batch(batch, mask_prob)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                outputs = model(masked_batch)
                mlm_logits = outputs["mlm_logits"]
                vocab_size = mlm_logits.size(-1)
                mlm_logits = mlm_logits.reshape(-1, vocab_size)
                mlm_labels_flat = mlm_labels.reshape(-1)
                
                mlm_loss = F.cross_entropy(
                    mlm_logits, 
                    mlm_labels_flat, 
                    ignore_index=-100
                )
                loss = mlm_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            
            # 日志记录
            if (step_idx + 1) % log_interval == 0:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - step_idx - 1)
                
                logger.info(
                    f"Epoch {ep+1}/{epochs} | Step {step_idx+1}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | MLM Loss: {mlm_loss.item():.4f} | "
                    f"ETA: {eta:.1f}s"
                )
        
        avg_loss = total_loss / steps_per_epoch
        avg_mlm_loss = total_mlm_loss / steps_per_epoch
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {ep+1} finished in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | Avg MLM Loss: {avg_mlm_loss:.4f}"
        )

def generate_embeddings(
    model: TransformerMLM,
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
        for step_idx, i in enumerate(range(0, total, batch_size)):
            batch_ids = client_ids[i: i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch)
            batch_embeddings = outputs["embed"].detach().cpu().numpy()
            
            end_idx = min(i + batch_size, total)
            embeddings[i:end_idx] = batch_embeddings.astype(np.float16)
            
            if (step_idx + 1) % log_interval == 0:
                processed = min(i + batch_size, total)
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-6)
                logger.info(
                    f"Embedding gen {step_idx+1}/{steps} | "
                    f"{processed}/{total} samples | {rate:.1f} samples/s"
                )
    
    return client_ids, embeddings

def main(params):
    # 初始化
    data_dir = DataDir(Path(params.data_dir))
    embeddings_dir = Path(params.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    
    # 加载数据
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    logger.info("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    logger.info("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)
    
    # 设备设置
    if params.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(params.device)
    
    use_amp = device.type == "cuda" and not params.no_amp
    
    # 创建模型
    model = TransformerMLM(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout
    )
    model.to(device)
    logger.info(f"Using device: {device} (AMP: {use_amp})")
    
    logger.info("Training Transformer MLM...")
    train_mlm(
        model=model,
        device=device,
        client_groups=client_groups,
        batch_size=params.batch_size,
        max_seq_len=params.max_seq_len,
        epochs=params.epochs,
        lr=params.lr,
        log_interval=params.log_interval,
        use_amp=use_amp,
        amp_device_type="cuda" if device.type == "cuda" else "cpu",
        mask_prob=params.mask_prob,
    )
    
    # 生成嵌入
    logger.info("Generating embeddings...")
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
    
    # 保存结果
    np.save(embeddings_dir / "client_ids.npy", client_ids)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    logger.info(f"Saved embeddings to {embeddings_dir}")

if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())