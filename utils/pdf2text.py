"""
This file is to preprocess PDF and XML files.
"""

import os
import sys
import typing
from typing import List
import argparse
from collections.abc import Callable
import numpy as np
import polars as pl # exploring instead of pandas
import pymupdf #does not extract tables


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2025"
__credits__ = ["Iwan"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__email__ = "ry222@umsystem.edu"
__status__ = "Development"
__date__ = "2025.02.28"


parser = argparse.ArgumentParser(
                                 description = "PDF to text conversion"
                                 )

parser.add_argument(
                    "--source",
                    "-s", 
                    type = str,
                    help = "The source containing .pdf files",
                    action = "store"
                    )

parser.add_argument(
                    "--destination",
                    "-d",
                    type = str,
                    help = "The destination directory for text files",
                    action = "store"
                    )

args = parser.parse_args()

try:
    DIR_SOURCE = args.source
    if not os.path.exists(DIR_SOURCE):
        raise FileNotFoundError('No folder exists at the location specified')
        
    DIR_DESTINATION = args.destination
    if not os.path.exists(DIR_DESTINATION):
        os.mkdir(DIR_DESTINATION)
except Exception as e:
    print(f"error occurred: {e}")
    

contents = os.listdir(DIR_SOURCE)
for article in contents:
    past_bibline = False
    article_name, extension = os.path.splitext(article)
    if extension == ".PDF" or extension == ".pdf":
        with open(os.path.join(DIR_DESTINATION, article_name + ".txt"), "w") as file:
            doc = pymupdf.open(os.path.join(DIR_SOURCE, article))
            for page in doc:
                text = page.get_text("blocks")
                for paragraph in text:
                    if isinstance(paragraph[4], str) and "References" in paragraph[4]:
                        past_bibline = True
                    if past_bibline:    
                        file.write(paragraph[4])
                    else:
                        file.write(paragraph[4].replace("\n", " "))
                        file.write("\n")
