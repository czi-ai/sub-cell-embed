from typing import Tuple

import torch
from torch import Tensor, nn


class AttentionPooler(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()

        self.out_dim = dim * num_heads

        self.attention = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, num_heads)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        attn = self.attention(x).permute(0, 2, 1)
        attn = self.softmax(attn)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        return x, attn


class GatedAttentionPooler(nn.Module):
    def __init__(
        self,
        dim: int,
        int_dim: int = 512,
        num_heads: int = 1,
        out_dim: int = None,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.attention_v = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(dim, int_dim), nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Dropout(self.dropout), nn.Linear(dim, int_dim), nn.GELU()
        )
        self.attention = nn.Linear(int_dim, num_heads)

        self.softmax = nn.Softmax(dim=-1)

        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)

        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)

        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)

        x = self.out_proj(x)
        return x, attn
