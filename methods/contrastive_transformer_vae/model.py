from __future__ import annotations

import torch
from torch import nn
from methods.contrastive_transformer.model import PositionalEncoding


class ContrastiveVAEModule(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        latent_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        fusion_weight: float = 0.5,
        recon_dim: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.fusion_weight = fusion_weight
        self.recon_dim = recon_dim
        # token embeddings (hashed buckets)
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)
        dim_ff = max(d_model, int(d_model * ffn_mult))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Contrastive projection head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, embed_dim)
        )
        # VAE heads
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim)
        hid = max(latent_dim * 2, 64)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, recon_dim),
        )

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_events(self, batch) -> torch.Tensor:
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
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        h = x[:, 0, :]
        return h

    def forward(self, batch, sample_latent: bool = True):
        h = self.encode_events(batch)
        proj = self.proj(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        recon = self.decoder(z)
        fused = proj + self.fusion_weight * self.latent_to_embed(mu)
        embed = nn.functional.normalize(fused, dim=-1)
        return {
            "embed": embed,
            "mu": mu,
            "logvar": logvar,
            "recon": recon,
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
