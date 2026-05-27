import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embedding_dim)

    def forward(self, x, attention_mask=None):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask shape: (B, S), 1 for real tokens and 0 for PAD.
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)

        output = self._merge_heads(context)
        return self.out_proj(output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embedding_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attn_out = self.attn(x, attention_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class MiniTransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        max_len=24,
        embedding_dim=64,
        num_heads=4,
        num_layers=1,
        ff_dim=128,
        hidden_dim=64,
        dropout=0.2,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embed_dropout(token_emb + pos_emb)

        attention_mask = (input_ids != self.pad_id).long()
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        mask = attention_mask.unsqueeze(-1).float()
        masked_sum = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked_sum / lengths

        return self.classifier(pooled)
