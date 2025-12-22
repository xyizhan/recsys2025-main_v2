# python -m baseline.aggregated_features_baseline.create_embeddings --data-dir /root/shared-nvme/data --embeddings-dir /root/shared-nvme/data/embedding/baseline
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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p

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
        return x + self.pe[:x.size(1)].unsqueeze(0)

class SimpleSASRec(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        
        sku_buckets = 65536
        cat_buckets = 4096
        url_buckets = 65536
        price_buckets = 128
        type_buckets = 8
        
        # 事件编码器
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        
        # [CLS] token
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.event_type_head = nn.Linear(d_model, 5)  # 只预测5个真实事件类型
        self.next_embed_head = nn.Linear(d_model, d_model)
        
        # 用户表征投影
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
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
        
        return self.ln(self.gelu(x))

    def create_causal_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, batch, return_predictions=False):
        x = self.encode_events(batch)
        B, L, _ = x.shape
        
        cls_token = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, L+1, D)
        x = self.dropout(self.pos(x))
        
        # 因果掩码
        causal_mask = None
        if self.training or return_predictions:
            causal_mask = self.create_causal_mask(L + 1).to(x.device)
        
        # Transformer编码
        x = self.transformer(x, mask=causal_mask)
        
        # [CLS]作为用户表征
        cls_output = x[:, 0, :]
        user_embed = self.proj(cls_output)
        user_embed = F.normalize(user_embed, dim=-1)
        
        if self.training or return_predictions:
            seq_output = x[:, 1:, :]  # (B, L, D)
            
            # 预测下一事件类型
            next_type_logits = self.event_type_head(seq_output)  # (B, L, 5)
            
            # 预测下一事件嵌入
            next_embed_pred = self.next_embed_head(seq_output)  # (B, L, D)
            
            return {
                'user_embed': user_embed,
                'next_type_logits': next_type_logits,
                'next_embed_pred': next_embed_pred,
                'hidden_states': seq_output
            }
        else:
            return {
                'user_embed': user_embed
            }


def compute_loss(model, model_outputs, batch, next_type_labels):
    seq_output = model_outputs['hidden_states']  # (B, L, D)
    next_type_logits = model_outputs['next_type_logits']  # (B, L, 5)
    next_embed_pred = model_outputs['next_embed_pred']  # (B, L, D)
    
    B, L, D = seq_output.shape
    type_loss = F.cross_entropy(
        next_type_logits.reshape(-1, 5),
        next_type_labels.reshape(-1),
        ignore_index=-100
    )
    
    # 嵌入预测损失：预测下一个位置的嵌入
    if L > 1:
        current_hidden = seq_output[:, :-1, :]  # (B, L-1, D)
        next_event_batch = {}
        for key in ['type_ids', 'sku_ids', 'cat_ids', 'price_ids', 'url_ids', 'query_vec']:
            if key in batch:
                next_event_batch[key] = batch[key][:, 1:]
        with torch.no_grad():
            next_embed_actual = model.encode_events(next_event_batch)  # (B, L-1, D)
        next_embed_pred_current = next_embed_pred[:, :-1, :]  # (B, L-1, D)
        mse_loss = F.mse_loss(next_embed_pred_current, next_embed_actual)
        cos_sim = F.cosine_similarity(next_embed_pred_current, next_embed_actual, dim=-1)
        cos_loss = 1 - cos_sim.mean()
        
        embed_loss = mse_loss + 0.5 * cos_loss
    else:
        embed_loss = torch.tensor(0.0, device=seq_output.device)
    
    # 总损失
    total_loss = type_loss + 0.3 * embed_loss  # 降低嵌入损失的权重
    
    return total_loss, type_loss.item(), embed_loss.item()


def prepare_labels(batch):
    type_ids = batch["type_ids"]
    B, L = type_ids.shape
    next_type_labels = torch.full((B, L), -100, dtype=torch.long, device=type_ids.device)
    
    if L > 1:
        next_type_ids = type_ids[:, 1:]
        valid_mask = (next_type_ids >= 2) & (next_type_ids <= 6)
        mapped_ids = next_type_ids - 2  # 映射到0-4
        next_type_labels[:, :L-1][valid_mask] = mapped_ids[valid_mask]  
    return next_type_labels


def train_model(
    model: SimpleSASRec,
    device: torch.device,
    client_groups,
    batch_size: int,
    max_seq_len: int,
    epochs: int,
    lr: float,
    log_interval: int,
    use_amp: bool,
    amp_device_type: str,
):
    ids = list(client_groups.keys())
    if len(ids) == 0:
        logger.warning("No clients available for training")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = max(1, math.ceil(len(ids) / batch_size))
    
    logger.info(f"Training on {len(ids)} clients ({steps_per_epoch} steps/epoch)")

    for ep in range(epochs):
        np.random.shuffle(ids)
        epoch_start = time.time()
        total_loss = 0
        total_type_loss = 0
        total_embed_loss = 0
        
        for step_idx in range(steps_per_epoch):
            # 准备批次数据
            start = step_idx * batch_size
            batch_ids = ids[start: start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 准备标签
            next_type_labels = prepare_labels(batch).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                # 前向传播
                outputs = model(batch, return_predictions=True)
                
                # 计算损失
                loss, type_loss, embed_loss = compute_loss(model, outputs, batch, next_type_labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_type_loss += type_loss
            total_embed_loss += embed_loss
            
            # 日志记录
            if (step_idx + 1) % log_interval == 0:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - step_idx - 1)
                
                logger.info(
                    f"Epoch {ep+1}/{epochs} | Step {step_idx+1}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | Type: {type_loss:.4f} | Embed: {embed_loss:.4f} | "
                    f"ETA: {eta:.1f}s"
                )
        
        avg_loss = total_loss / steps_per_epoch
        avg_type_loss = total_type_loss / steps_per_epoch
        avg_embed_loss = total_embed_loss / steps_per_epoch
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {ep+1} finished in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | Type: {avg_type_loss:.4f} | Embed: {avg_embed_loss:.4f}"
        )

def generate_embeddings(
    model: SimpleSASRec,
    device: torch.device,
    client_groups,
    relevant_client_ids: np.ndarray,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    log_interval: int,
):
    """生成用户嵌入"""
    model.eval()
    client_ids = relevant_client_ids.astype(np.int64)
    embeddings = np.zeros((client_ids.shape[0], embedding_dim), dtype=np.float16)
    
    total = client_ids.shape[0]
    steps = max(1, math.ceil(total / batch_size))
    
    start_time = time.time()
    
    with torch.no_grad():
        for step_idx, i in enumerate(range(0, total, batch_size)):
            batch_ids = client_ids[i: i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch, return_predictions=False)
            batch_embeddings = outputs["user_embed"].detach().cpu().numpy()
            
            end_idx = min(i + batch_size, total)
            embeddings[i:end_idx] = batch_embeddings.astype(np.float16)
            
            if (step_idx + 1) % log_interval == 0:
                processed = end_idx
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
    model = SimpleSASRec(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        n_layers=params.n_layers,
        n_heads=params.n_heads,
        dropout=params.dropout,
    )
    model.to(device)
    
    logger.info(f"Using device: {device} (AMP: {use_amp})")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    logger.info("Training Simple SASRec...")
    train_model(
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
    logger.info(f"Saved embeddings for {len(client_ids)} clients to {embeddings_dir}")
    logger.info(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())