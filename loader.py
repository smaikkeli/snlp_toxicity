import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from encoder.py import Encoder
from torch.nn.utils.rnn import pad_sequence


import warnings
warnings.filterwarnings("ignore")

def tokenize_and_encode(text, vocabulary):
    """Tokenizes and encodes a text sequence into a list of integers.

    Args:
        text (str): The text sequence to encode.
        vocabulary (dict): A dictionary mapping tokens to unique integers.

    Returns:
        List of integers representing the encoded text.
    """
    tokens = text.split()
    return [vocabulary.get(token, vocabulary['<UNK>']) for token in tokens]

class ToxicityDataset(Dataset):
    """Toxicity dataset."""

    def __init__(self, filename, id_col, text_col, label_col, vocabulary):
        """
        Arguments:
            id - column for the example_id within the set
            text - the text of the comment
            label - binary label (1=Toxic/0=NonToxic)
        """
        self.dataframe = pd.read_csv(filename)
        self.ids = self.dataframe[id_col].values
        self.texts = [tokenize_and_encode(text, vocabulary) for text in self.dataframe[text_col].values]
        self.labels = self.dataframe[label_col].values
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]



def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (src_seq, tgt_seq):
          src_seq is of shape (src_seq_length,)
          tgt_seq is of shape (tgt_seq_length,)

    Returns:
      src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
          The sequences should be sorted by length in a decreasing order, that is src_seqs[:,0] should be
          the longest sequence, and src_seqs[:,-1] should be the shortest.
      src_seq_lengths: List of lengths of source sequences.
      tgt_seqs of shape (max_tgt_seq_length, batch_size): Tensor of padded target sequences.
    """
    # YOUR CODE HERE
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Extract source and target sequences
    src_seqs, tgt_seqs = zip(*list_of_samples)
    
    # Pad source sequences
    src_seqs_padded = pad_sequence(src_seqs, padding_value=0)
    
    # Pad target sequences
    tgt_seqs_padded = pad_sequence(tgt_seqs, padding_value=0)
    
    # Get lengths of source sequences
    src_seq_lengths = [len(seq) for seq in src_seqs]
    
    return src_seqs_padded, src_seq_lengths, tgt_seqs_padded

    
#train_dataset = ToxicityDataset(filename='data/train_2024.csv')

#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate)

