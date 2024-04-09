import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden = 1024, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(n_features, n_heads, batch_first = True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_features)
        )
        self.norm1 = nn.LayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x2, _ = self.attn(x, x, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.feed_forward(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

def clones(module, N):
    "Produces N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_blocks = 4, n_features = 256, n_heads = 16, n_hidden=64, dropout=0.1, max_length = 5000):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, n_features)
        self.pos_embedding = nn.Embedding(max_length, n_features)
        self.blocks = nn.ModuleList([EncoderBlock(n_features, n_heads, n_hidden, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(n_features)
        
    def forward(self, x, mask):
        B, T = x.size()
        positions = torch.arange(0, T, device = device)
        x = self.embedding(x)
        x = x + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)
    
#Classifier on top of the encoder
class MLPClassifier(nn.Module):
    def __init__(self, n_features=512, num_classes=2, num_layers=3, dropout=0.2):
        super(MLPClassifier, self).__init__()
        
        #A single layer
        layers = [
            nn.Linear(n_features, n_features*4),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        #Append all layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(n_features*4, n_features*4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        #The output layer
        if num_classes == 2:
            layers.append(nn.Linear(n_features*4, 1))
        else:
            layers.append(nn.Linear(n_features*4, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
    

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        
        #Take the [CLS] token
        x = x[:,   0, :]

        #x = x.mean(dim = 1)
        
        x = self.classifier(x)
        return x