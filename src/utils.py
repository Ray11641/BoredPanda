"""
This is file provides utilities for preprocessing and post processing 

"""

import os
import numpy as np


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2025"
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__status__ = "Development"
__date__ = "2025.06.28"


POS_LOOKUP = {
            "ADJ": "a",
            "ADP": "n",
            "ADV": "r",
            "CONJ": "n",
            "DET": "n",
            "NOUN": "n",
            "NUM": "n",
            "PRT": "n",
            "PRON": "n",
            "VERB": "v",
            ".": "n",
            "X": "n"
                }