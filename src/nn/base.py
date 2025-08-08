import typing
from typing import List
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
                  n_embed: int,
                    n_head: int) -> None:
        super().__init__()
        self.Q = nn.Linear(n_embed, n_head)
        self.K = nn.Linear(n_embed, n_head)
        self.V = nn.Linear(n_embed, n_head)

    def forward(self, 
                h: torch.Tensor) -> List[torch.Tensor]





class base_nn(nn.Module):
    """
    create a base class for constructing custom transformer
    
    """
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def encoder(cls):
        pass  

    @classmethod
    def decoder(cls):
        pass


    