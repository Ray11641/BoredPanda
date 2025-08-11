# BoredPanda

This repository is to save all my implementations of natural language processing tools, mostly using low-embedding size transformer models. Particularly, BoredPanda project is aimed to create a bot that gives a summary to each of the presented pdf document. 

![alt text](https://github.com/Ray11641/BoredPanda/blob/04d46c074f7f3541084975276e649ccbc0c1b452/BoredPanda_Logo.png "BoredPanda")

## Please Note!
- Feel free to use the code, it's public for a reason!
- If you notice bugs, buy me a bugspray by reporting them.
- This code attempts to implement full transformer; for some reason most repos stop with encoder. 
- While using larger embedding sizes does offer more parameteric complexity, thus larger hypothesis space, I do not have a GPU and I am limited to Google Colab; hence, chose low embedding size. 
- The implementation of encoder and decoder modules use prelayer normalisation, thus the encoder output presented to decoder is internally normalised.
- The model is going to be trained to generate the abstracts for presented articles, abstracts are considered summary here. 

### Disclaimer
#1: PDFs are downloaded from Arxiv. I do not own any articles other than my own. Hence I am not going to upload them to this repository.
#2: Neural Network training process is going to be extremely slow, if you need weights (parameters), you hafta wait!
#3: I used Tensorflow and Chainer[^1], and not PyTorch. This technically is first large project with PyTorch. I am using PyTorch due to Chainer, and might write the implementations in Chainer.
[^1]: If you don't know Chainer, you should check about it! The Define-by-run is because of them and the library is awesome. 

## Requirements

Needs the following python libraries:

> UV

use UV to install the requirements and create an environment. However, if you use GPU, change the _pyproject.toml_ to download PyTorch-GPU. 

