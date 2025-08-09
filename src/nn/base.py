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
                h: torch.Tensor) -> List[torch.Tensor]:
        attn_outs = self.scaled_dot_product_attention(
                                                      Q = self.Q(h),
                                                      K = self.K(h),
                                                      V = self.V(h)
                                                     )
        return attn_outs

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
                h: torch.Tensor) -> torch.Tensor:
        multihead_out = torch.cat([head(h) for head in self.heads], dim = -1)
        return self.multihead_linear(multihead_out)

class base_nn(nn.Module):
    """f
    create a base class for constructing custom transformer
    
    """
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def encoder(cls):
        

    @classmethod
    def decoder(cls):
        pass


    