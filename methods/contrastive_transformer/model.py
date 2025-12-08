import torch
from torch import nn


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


class EventEncoder(nn.Module):
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
        ffn_mult: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        # token embeddings (hashed buckets)
        self.emb_type = nn.Embedding(type_buckets, d_model)
        self.emb_sku = nn.Embedding(sku_buckets, d_model)
        self.emb_cat = nn.Embedding(cat_buckets, d_model)
        self.emb_url = nn.Embedding(url_buckets, d_model)
        self.emb_price = nn.Embedding(price_buckets, d_model)
        # query vector (length 16 integers) projection
        self.query_proj = nn.Linear(16, d_model)
        # special tokens
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
        # projection head (SimCLR style)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, embed_dim)
        )
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _hash(self, ids: torch.Tensor, buckets: int) -> torch.Tensor:
        return (ids % buckets).clamp(min=0)

    def event_embed(
        self,
        type_ids: torch.Tensor,
        sku_ids: torch.Tensor | None,
        cat_ids: torch.Tensor | None,
        price_ids: torch.Tensor | None,
        url_ids: torch.Tensor | None,
        query_vec: torch.Tensor | None,
    ) -> torch.Tensor:
        # all tensors shape: (B, L)
        x = self.emb_type(self._hash(type_ids, self.emb_type.num_embeddings))
        if sku_ids is not None:
            x = x + self.emb_sku(self._hash(sku_ids, self.emb_sku.num_embeddings))
        if cat_ids is not None:
            x = x + self.emb_cat(self._hash(cat_ids, self.emb_cat.num_embeddings))
        if price_ids is not None:
            x = x + self.emb_price(self._hash(price_ids, self.emb_price.num_embeddings))
        if url_ids is not None:
            x = x + self.emb_url(self._hash(url_ids, self.emb_url.num_embeddings))
        if query_vec is not None:
            x = x + self.query_proj(query_vec)
        return self.ln(self.gelu(x))

    def forward(self, batch):
        # batch dict: {type_ids, sku_ids?, cat_ids?, price_ids?, url_ids?, query_vec?} shapes (B, L)
        x = self.event_embed(
            type_ids=batch["type_ids"],
            sku_ids=batch.get("sku_ids"),
            cat_ids=batch.get("cat_ids"),
            price_ids=batch.get("price_ids"),
            url_ids=batch.get("url_ids"),
            query_vec=batch.get("query_vec"),
        )
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.pos(x))
        x = self.encoder(x)
        # take CLS token
        h = x[:, 0, :]
        z = self.proj(h)
        z = nn.functional.normalize(z, dim=-1)
        return z


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # z1, z2: (B, D) normalized
    logits = (z1 @ z2.t()) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    loss1 = nn.functional.cross_entropy(logits, targets)
    loss2 = nn.functional.cross_entropy(logits.t(), targets)
    return (loss1 + loss2) * 0.5
