"""
This is base class for a bot 

"""

import os
import random
import typing
from typing import List

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .utils import *

__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2025"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__status__ = "Development"
__date__ = "2025.06.28"


class BaseBot():
    """
    Base bot class
    """
    def __init__(self, nn_model):
        """
        
        """
        self.nn_model = nn_model
        
    @staticmethod
    def tokenize_and_lemmatize(S: str) -> List[str]:
        """
        Arguments
        :param S: should be an individual sentence
        
        Returns
        :param S_lemmas: the lemmas in the sentence
        """
        tokens = nltk.word_tokenize(S)
        tags = nltk.pos_tag(tokens, tagset = "universal")
        
        lemmatizer = nltk.WordNetLemmatizer()
        S_lemmas = [lemmatizer.lemmatize(word.lower(), pos = POS_LOOKUP[part_speech]) for word, part_speech in tags]
        
        return S_lemmas
    
    # @staticmethod
    # def 