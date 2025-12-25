from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from methods.contrastive_transformer.model import PositionalEncoding


class ShallowInterestVAE(nn.Module):
    """Lightweight VAE that focuses on recent/local context."""

    def __init__(self, input_dim: int, latent_dim: int, recon_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(128, input_dim // 2)
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        dec_hidden = max(latent_dim * 2, hidden_dim)
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
        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "recon": recon,
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class DeepInterestVAE(nn.Module):
    """Hierarchical VAE that models slow-moving preferences."""

    def __init__(self, input_dim: int, latent_dim: int, recon_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(256, input_dim)
        self.latent_dim = latent_dim
        self.global_dim = max(8, latent_dim // 2)
        self.local_dim = max(8, latent_dim - self.global_dim)
        enc_dim = max(hidden_dim, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
        )
        self.global_mu = nn.Linear(enc_dim, self.global_dim)
        self.global_logvar = nn.Linear(enc_dim, self.global_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(enc_dim + self.global_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
        )
        self.local_mu = nn.Linear(enc_dim, self.local_dim)
        self.local_logvar = nn.Linear(enc_dim, self.local_dim)
        dec_hidden = max(hidden_dim, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden),
            nn.GELU(),
            nn.Linear(dec_hidden, dec_hidden),
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
        base = self.encoder(x)
        g_mu = self.global_mu(base)
        g_logvar = self.global_logvar(base)
        g_latent = self.reparameterize(g_mu, g_logvar, sample_latent)
        local_hidden = self.local_encoder(torch.cat([base, g_latent], dim=-1))
        l_mu = self.local_mu(local_hidden)
        l_logvar = self.local_logvar(local_hidden)
        l_latent = self.reparameterize(l_mu, l_logvar, sample_latent)
        latent = torch.cat([g_latent, l_latent], dim=-1)
        mu = torch.cat([g_mu, l_mu], dim=-1)
        logvar = torch.cat([g_logvar, l_logvar], dim=-1)
        recon = self.decoder(latent)
        level_gap = torch.relu(l_mu.abs().mean(dim=-1) - g_mu.abs().mean(dim=-1))
        var_gap = torch.relu(g_logvar.mean(dim=-1) - l_logvar.mean(dim=-1))
        hierarchy_loss = torch.mean(level_gap + 0.1 * var_gap)
        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "recon": recon,
            "hierarchy_loss": hierarchy_loss,
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class DualCLSContrastiveTransformer(nn.Module):
    """Transformer encoder with dual CLS tokens feeding shallow/deep VAEs."""

    def __init__(
        self,
        d_model: int = 256,
        embed_dim: int = 512,
        shallow_latent_dim: int = 96,
        deep_latent_dim: int = 160,
        recon_dim_shallow: int = 5,
        recon_dim_deep: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        sku_buckets: int = 65536,
        cat_buckets: int = 4096,
        url_buckets: int = 65536,
        price_buckets: int = 128,
        type_buckets: int = 8,
        dropout: float = 0.1,
        ffn_mult: float = 2.0,
        short_window: int = 32,
        stats_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.short_window = max(1, short_window)
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        self.query_proj = nn.Linear(16, d_model)
        self.cls_short = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_long = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.short_stats_proj = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
        )
        self.long_stats_proj = nn.Sequential(
            nn.LayerNorm(stats_dim),
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model=d_model)
        self.ln = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
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
        self.shallow_vae = ShallowInterestVAE(
            input_dim=d_model * 3,
            latent_dim=shallow_latent_dim,
            recon_dim=recon_dim_shallow,
            hidden_dim=d_model,
        )
        self.deep_vae = DeepInterestVAE(
            input_dim=d_model * 4,
            latent_dim=deep_latent_dim,
            recon_dim=recon_dim_deep,
            hidden_dim=d_model * 2,
        )
        self.shallow_to_embed = nn.Linear(shallow_latent_dim, embed_dim)
        self.deep_to_embed = nn.Linear(deep_latent_dim, embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        self.mask_predictor = nn.Linear(d_model, type_buckets)
        order_hidden = max(64, d_model)
        self.order_classifier = nn.Sequential(
            nn.Linear(d_model * 2, order_hidden),
            nn.GELU(),
            nn.Linear(order_hidden, 2),
        )
        decay = torch.linspace(1.0, 0.2, steps=self.short_window)
        self.register_buffer("short_decay", decay, persistent=False)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def _event_embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def encode_events(self, batch: dict[str, torch.Tensor], return_sequence: bool = False) -> torch.Tensor | tuple[torch.Tensor, ...]:
        stats_vec = batch["stats_vec"]
        stats_short = self.short_stats_proj(stats_vec)
        stats_long = self.long_stats_proj(stats_vec)
        event_emb = self._event_embed(batch)
        B = event_emb.size(0)
        cls_short = self.cls_short.expand(B, -1, -1) + stats_short.unsqueeze(1)
        cls_long = self.cls_long.expand(B, -1, -1) + stats_long.unsqueeze(1)
        tokens = torch.cat([cls_short, cls_long, event_emb], dim=1)
        tokens = self.dropout(self.pos(tokens))
        seq = self.encoder(tokens)
        if return_sequence:
            return seq
        sequence_tokens = seq[:, 2:, :]
        short_token = seq[:, 0, :]
        long_token = seq[:, 1, :]
        return short_token, long_token, sequence_tokens, stats_short, stats_long

    def _recent_context(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        B, L, _ = sequence_tokens.size()
        if L == 0:
            return torch.zeros(B, self.d_model, device=sequence_tokens.device)
        win = min(L, self.short_window)
        recent = sequence_tokens[:, -win:, :]
        weights = self.short_decay[:win].to(sequence_tokens.device)
        weights = weights / weights.sum().clamp(min=1e-6)
        context = torch.sum(recent * weights.view(1, win, 1), dim=1)
        return context

    def _global_context(self, sequence_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = sequence_tokens.size()
        if L == 0:
            zeros = torch.zeros(B, self.d_model, device=sequence_tokens.device)
            return zeros, zeros
        global_pool = sequence_tokens.mean(dim=1)
        trend = sequence_tokens[:, -1, :] - sequence_tokens[:, 0, :]
        return global_pool, trend

    def forward(self, batch: dict[str, torch.Tensor], sample_latent: bool = True) -> dict[str, torch.Tensor]:
        short_token, long_token, seq_tokens, stats_short, stats_long = self.encode_events(batch)
        recent_ctx = self._recent_context(seq_tokens)
        global_ctx, trend_ctx = self._global_context(seq_tokens)
        shallow_input = torch.cat([short_token, recent_ctx, stats_short], dim=-1)
        deep_input = torch.cat([long_token, global_ctx, trend_ctx, stats_long], dim=-1)
        shallow_out = self.shallow_vae(shallow_input, sample_latent=sample_latent)
        deep_out = self.deep_vae(deep_input, sample_latent=sample_latent)
        shallow_embed = self.shallow_to_embed(shallow_out["latent"])
        deep_embed = self.deep_to_embed(deep_out["latent"])
        gate_inp = torch.cat([shallow_embed, deep_embed], dim=-1)
        gate = self.fusion_gate(gate_inp)
        fused = gate * shallow_embed + (1 - gate) * deep_embed
        embed = F.normalize(fused, dim=-1)
        deep_target = torch.cat([global_ctx, stats_long], dim=-1)
        return {
            "embed": embed,
            "shallow_mu": shallow_out["mu"],
            "shallow_logvar": shallow_out["logvar"],
            "shallow_recon": shallow_out["recon"],
            "deep_mu": deep_out["mu"],
            "deep_logvar": deep_out["logvar"],
            "deep_recon": deep_out["recon"],
            "deep_target": deep_target,
            "hierarchy_loss": deep_out["hierarchy_loss"],
        }

    def predict_mask_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = self.encode_events(batch, return_sequence=True)
        tokens = seq[:, 2:, :]
        if tokens.size(1) == 0:
            tokens = torch.zeros(seq.size(0), 1, self.d_model, device=seq.device)
        return self.mask_predictor(tokens)

    def classify_order(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = self.encode_events(batch, return_sequence=True)
        short_token = seq[:, 0, :]
        long_token = seq[:, 1, :]
        pooled = torch.cat([short_token, long_token], dim=-1)
        return self.order_classifier(pooled)
