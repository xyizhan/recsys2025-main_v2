# hstu_baseline.py
import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
from torchinfo import summary

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
    p.add_argument("--num-blocks", type=int, default=2, help="Number of HSTU blocks")
    p.add_argument("--num-heads", type=int, default=1, help="Number of attention heads")
    p.add_argument("--dqk", type=int, default=64, help="Attention dimension")
    p.add_argument("--dv", type=int, default=64, help="Linear dimension")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p

class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    def __init__(self, max_seq_len: int, num_buckets: int = 128):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        
        # 位置偏置
        self.pos_w = nn.Parameter(torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02))
        # 时间偏置
        self.ts_w = nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0, std=0.02))
    
    def bucketize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long(),
            min=0,
            max=self.num_buckets
        )
    
    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        B, N = all_timestamps.shape
        
        # 位置偏置
        t = F.pad(self.pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        rel_pos_bias = t[:, :, r:-r]
        
        # 时间偏置
        ext_timestamps = torch.cat([all_timestamps, all_timestamps[:, N-1:N]], dim=1)
        time_diffs = ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
        bucketed_timestamps = self.bucketize(time_diffs)
        
        rel_ts_bias = self.ts_w[bucketed_timestamps.view(-1)].view(B, N, N)
        
        return rel_pos_bias + rel_ts_bias

class SequentialTransductionUnit(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        rel_attn_bias: Optional[RelativeBucketedTimeAndPositionBasedBias] = None,
        concat_ua: bool = False,
        epsilon: float = 1e-6
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.rel_attn_bias = rel_attn_bias
        self.concat_ua = concat_ua
        self.epsilon = epsilon
        
        # UVQK线性变换
        self.uvqk = nn.Linear(
            embedding_dim,
            linear_dim * 2 * num_heads + attention_dim * num_heads * 2
        )
        
        output_dim = linear_dim * num_heads * (3 if concat_ua else 1)
        self.o = nn.Linear(output_dim, embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim, eps=epsilon)
        self.ln2 = nn.LayerNorm(linear_dim * num_heads, eps=epsilon)
        self.dropout_layer = nn.Dropout(dropout)
        self.attn_dropout_layer = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        x_norm = self.ln1(x)
        uvqk_out = self.uvqk(x_norm)
        split_sizes = [
            self.linear_dim * self.num_heads,
            self.linear_dim * self.num_heads,
            self.attention_dim * self.num_heads,
            self.attention_dim * self.num_heads
        ]
        u, v, q, k = torch.split(uvqk_out, split_sizes, dim=-1)
        q = q.view(B, N, self.num_heads, self.attention_dim)           # (B, N, H, dqk)
        k = k.view(B, N, self.num_heads, self.attention_dim)           # (B, N, H, dqk)
        v = v.view(B, N, self.num_heads, self.linear_dim)              # (B, N, H, dv)
        
        q = q.transpose(1, 2)  # (B, H, N, dqk)
        k = k.transpose(1, 2)  # (B, H, N, dqk)
        v = v.transpose(1, 2)  # (B, H, N, dv)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_dim)  # (B, H, N, N)
        if self.rel_attn_bias is not None and timestamps is not None:
            rel_bias = self.rel_attn_bias(timestamps)  # (B, N, N)
            attn_scores = attn_scores + rel_bias.unsqueeze(1)  # (B, H, N, N)

        attn_weights = F.silu(attn_scores) / float(N)  # (B, H, N, N)
        attn_weights = attn_weights * attention_mask

        attn_weights = self.attn_dropout_layer(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, dv)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N, H, dv)
        attn_output = attn_output.view(B, N, self.linear_dim * self.num_heads)  # (B, N, H*dv)
        a = self.ln2(attn_output)

        if self.concat_ua:
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * a
        
        o_input = self.dropout_layer(o_input)
        output = self.o(o_input)
        
        return output + x

class EmbeddingModule(nn.Module):
    def __init__(self, d_model: int = 256, item_embedding_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.item_embedding_dim = item_embedding_dim
        
        sku_buckets = 65536
        cat_buckets = 4096
        url_buckets = 65536
        price_buckets = 128
        type_buckets = 8
        
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        
    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)
    
    def get_item_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=math.sqrt(1.0 / embedding_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        pos_embeddings = self.pos_emb(positions)
        return self.dropout(x * math.sqrt(D) + pos_embeddings)

class HSTUBaseline(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        num_blocks: int = 2,
        num_heads: int = 1,
        dqk: int = 64,
        dv: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        concat_ua: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding_module = EmbeddingModule(d_model=d_model)
        self.pos_encoder = LearnablePositionalEmbedding(max_seq_len, d_model, dropout)
        
        # HSTU
        self.blocks = nn.ModuleList([
            SequentialTransductionUnit(
                embedding_dim=d_model,
                linear_dim=dv,
                attention_dim=dqk,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=dropout,
                rel_attn_bias=RelativeBucketedTimeAndPositionBasedBias(max_seq_len),
                concat_ua=concat_ua
            )
            for _ in range(num_blocks)
        ])
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim)
        )
        
        # 因果注意力掩码
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1
        ))
    
    def create_attention_mask(self, batch_size: int, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        return mask.expand(batch_size, 1, seq_len, seq_len)
    
    def forward(self, batch, return_predictions=False):
        item_embeddings = self.embedding_module.get_item_embeddings(batch)  # (B, L, D)
        x = self.pos_encoder(item_embeddings)
        B, L, _ = x.shape
        attention_mask = self.create_attention_mask(B, L).to(x.device)
        timestamps = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L).float()

        for block in self.blocks:
            x = block(x, attention_mask, timestamps)

        user_repr = x[:, -1, :] if L > 0 else x.mean(dim=1)
        user_embed = self.proj(user_repr)
        user_embed = F.normalize(user_embed, dim=-1) 
        return {"user_embed": user_embed}

def train_hstu(
    model: HSTUBaseline,
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
        
        for step_idx in range(steps_per_epoch):
            start = step_idx * batch_size
            batch_ids = ids[start: start + batch_size]
            seqs = [client_groups[cid] for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            batch2 = {k: v.clone() for k, v in batch.items()}
            if "type_ids" in batch2:
                mask_prob = 0.1
                mask = torch.rand(batch2["type_ids"].shape, device=batch2["type_ids"].device) < mask_prob
                batch2["type_ids"][mask] = TYPE_TO_ID.get("MASK", 0)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                outputs1 = model(batch)
                outputs2 = model(batch2)
                z1 = outputs1["user_embed"]
                z2 = outputs2["user_embed"]
                
                # InfoNCE损失
                temperature = 0.2
                logits = (z1 @ z2.t()) / temperature
                targets = torch.arange(z1.size(0), device=z1.device)
                loss1 = F.cross_entropy(logits, targets)
                loss2 = F.cross_entropy(logits.t(), targets)
                loss = (loss1 + loss2) * 0.5

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (step_idx + 1) % log_interval == 0:
                elapsed = time.time() - epoch_start
                avg_step = elapsed / (step_idx + 1)
                eta = avg_step * (steps_per_epoch - step_idx - 1)
                
                logger.info(
                    f"Epoch {ep+1}/{epochs} | Step {step_idx+1}/{steps_per_epoch} | "
                    f"Loss: {loss.item():.4f} | ETA: {eta:.1f}s"
                )
        
        avg_loss = total_loss / steps_per_epoch
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {ep+1} finished in {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}"
        )

def generate_embeddings(
    model: HSTUBaseline,
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
    
    start_time = time.time()
    
    with torch.no_grad():
        for step_idx, i in enumerate(range(0, total, batch_size)):
            batch_ids = client_ids[i: i + batch_size]
            seqs = [client_groups.get(int(cid), None) for cid in batch_ids]
            batch = collate_sequences(seqs, max_len=max_seq_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch)
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
    data_dir = DataDir(Path(params.data_dir))
    embeddings_dir = Path(params.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    
    relevant_client_ids = np.load(data_dir.input_dir / "relevant_clients.npy")
    logger.info("Loading input events...")
    dfs = load_events_df(data_dir=data_dir)
    logger.info("Building client sequences...")
    client_groups = build_client_sequences(dfs=dfs, relevant_client_ids=relevant_client_ids)

    if params.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(params.device)
    
    use_amp = device.type == "cuda" and not params.no_amp
    
    model = HSTUBaseline(
        d_model=params.d_model,
        embed_dim=params.embedding_dim,
        num_blocks=params.num_blocks,
        num_heads=params.num_heads,
        dqk=params.dqk,
        dv=params.dv,
        dropout=params.dropout,
        max_seq_len=params.max_seq_len,
    )
    model.to(device)
    
    logger.info(f"Using device: {device} (AMP: {use_amp})")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # summary(model, input_size=(1, 256, 256))

    # 训练
    logger.info("Training HSTU...")
    train_hstu(
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