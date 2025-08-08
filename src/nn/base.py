import typing
import torch
import torch.nn as nn

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


    