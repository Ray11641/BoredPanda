import typing
from typing import List, Dict
import math
import torch
import torch.nn as nn


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2025"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__status__ = "Development"
__date__ = "2025.06.28"
__all__ = ["base_nn"]


class AttentionHead(nn.Module):
    """
    A simple AttentionHead
    """
    def __init__(self,
                 d_embed: int,
                 d_head: int) -> None:
        super().__init__()
        self.Q = nn.Linear(d_embed, d_head)
        self.K = nn.Linear(d_embed, d_head)
        self.V = nn.Linear(d_embed, d_head)

    def scaled_dot_product_attention(Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor
                                     ) -> torch.Tensor:
        """
        Compute the scaled dot product attention for given query (Q), and
          key-value (K-V) pairs
        """
        n_k = Q.shape(-1)
        prod = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(n_k)
        weights = nn.functional.softmax(prod, dim = -1)
        return torch.bmm(weights, V)

    def forward(self, 
                x: torch.Tensor) -> List[torch.Tensor]:
        x = self.scaled_dot_product_attention(
                                              Q = self.Q(x),
                                              K = self.K(x),
                                              V = self.V(x)
                                             )
        return x

class MultiHeadAttention(nn.Module):
    """
    Uses Attention Head for computation
    """
    def __init__(self,
                 d_embed: int,
                 d_head: int,
                 n_heads: int) -> None:
        super()._init__()
        # raise error if d_embed != d_head*n_heads
        assert d_embed == d_head*n_heads
        self.heads = nn.ModuleList([AttentionHead(d_embed, d_head) for _ in range(n_heads)])
        self.multihead_linear = nn.Linear(d_embed, d_embed)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim = -1)
        return self.multihead_linear(x)

class FeedForward(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 droput_prob: float) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_in)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(droput_prob)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return self.dropout(x)

class base_nn(nn.Module):
    """f
    create a base class for constructing custom transformer
    
    """
    def __init__(self,
                 name: str) -> None:
        super().__init__()
        self.name = name

    @classmethod
    def encoder(cls, config: Dict[int]):
        e = cls(config["name"])
        e.norm_1 = nn.LayerNorm(config["hidden_size"])
        e.norm_2 = nn.LayerNorm(config["hidden_size"])
        e.attention = MultiHeadAttention(config["d_embed"],
                                         config["d_head"],
                                         config["n_heads"])
        e.feedforward = FeedForward(config["d_embed"],
                                    )

    @classmethod
    def decoder(cls):
        pass


    