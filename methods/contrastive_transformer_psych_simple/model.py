from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from methods.contrastive_transformer.model import PositionalEncoding


class SingleVAESummary(nn.Module):
    """Standard VAE used to capture fused sequential + handcrafted features."""

    def __init__(self, input_dim: int, latent_dim: int, recon_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        dec_hidden = max(hidden_dim, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden),
            nn.GELU(),
            nn.Linear(dec_hidden, recon_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, sample_latent: bool) -> torch.Tensor:
        if not sample_latent:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        latent = self.reparameterize(mu, logvar, sample_latent)
        recon = self.decoder(latent)
        return {"latent": latent, "mu": mu, "logvar": logvar, "recon": recon}

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class SingleVAEPsychTransformer(nn.Module):
    """Contrastive transformer encoder with one VAE head."""

    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        latent_dim: int = 160,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        stats_dim: int = 46,
        recon_dim: int = 5,
        stats_version: str = "concat",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.stats_proj = nn.Sequential(nn.BatchNorm1d(stats_dim), nn.Linear(stats_dim, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
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
        self.stats_version = stats_version
        if stats_version == "concat":
            self.h_fusion = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
            )
        elif stats_version == "gate":
            self.stats_to_film = nn.Linear(d_model, d_model * 2)
        self.vae = SingleVAESummary(input_dim=d_model, latent_dim=latent_dim, recon_dim=recon_dim, hidden_dim=max(d_model, 256))
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim)
        self.mask_predictor = nn.Linear(d_model, type_buckets)
        order_hidden = max(64, d_model // 2)
        self.order_classifier = nn.Sequential(
            nn.Linear(d_model, order_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(order_hidden, 2),
        )

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def encode_events(self, batch: dict[str, torch.Tensor], return_sequence: bool = False):
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
        seq = self.encoder(x)
        if return_sequence:
            return seq
        stats_emb = self.stats_proj(batch["stats_vec"])
        h = seq[:, 0, :]
        if self.stats_version == "concat":
            h = self.h_fusion(torch.cat([h, stats_emb], dim=-1))
        elif self.stats_version == "gate":
            gamma_beta = self.stats_to_film(stats_emb)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.sigmoid(gamma)
            beta = torch.tanh(beta) * 0.5
            h = h * gamma + beta
        return h, stats_emb

    def forward(self, batch: dict[str, torch.Tensor], sample_latent: bool = True) -> dict[str, torch.Tensor]:
        h, _ = self.encode_events(batch)
        vae_out = self.vae(h, sample_latent=sample_latent)
        embed = self.latent_to_embed(vae_out["mu"])
        embed = F.normalize(embed, dim=-1)
        return {
            "embed": embed,
            "vae_latent": vae_out["latent"],
            "vae_mu": vae_out["mu"],
            "vae_logvar": vae_out["logvar"],
            "vae_recon": vae_out["recon"],
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return SingleVAESummary.kl_divergence(mu, logvar)

    def predict_mask_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = self.encode_events(batch, return_sequence=True)
        tokens = seq[:, 1:, :]
        if tokens.size(1) == 0:
            tokens = torch.zeros(seq.size(0), 1, self.d_model, device=seq.device)
        return self.mask_predictor(tokens)

    def classify_order(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h, _ = self.encode_events(batch)
        return self.order_classifier(h)
