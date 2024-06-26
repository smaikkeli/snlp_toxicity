import os
import torch
import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import warnings
warnings.filterwarnings("ignore")

CLS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
MAX_LENGTH = 10

class BPE:
    def __init__(self, name):
        self.name = name
        self.vocab = {}

class Lang:
    """A class that encodes words with one-hot vectors."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "CLS", 1: "EOS"}  
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indicesToString(self, indices):
        """Converts a list of indices back to a string. Special tokens are handled."""
        words = []
        for idx in indices:
            word = self.index2word[idx.item()] if idx.item() in self.index2word else '[UNK]'
            if word == "CLS":
                words.append('[' + word + ']')
            else:
                words.append(word)
        return ' '.join(words)
    
    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "CLS", 1: "EOS"}  
        self.n_words = 2

        for word in keep_words:
            self.addWord(word)
    
    
def translateDatasetEntries(dataset, lang):
    translated_texts = []
    labels = []

    for i in range(len(dataset)):
        text_tensor, label_tensor = dataset[i] 
        text_indices = text_tensor.numpy() 

        translated_text = lang.indicesToString(text_indices)
        translated_texts.append(translated_text)
        
        labels.append(label_tensor.item())

    return translated_texts, labels


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


class ToxicityDataset(Dataset):
    """Toxicity dataset."""

    def __init__(self, filename, id_col, text_col, label_col, lang):
        self.data = pd.read_csv(filename, quoting=3)
        #If label only contains ?, then set it to -1
        self.data[label_col] = self.data[label_col].apply(lambda x: -1 if x == '?' else x)
        self.data[text_col] = self.data[text_col].apply(lambda x: unicodeToAscii(x))
        self.data[text_col] = self.data[text_col].apply(lambda x: normalizeString(x))

        self.lang = lang
        self.texts = [self.encode_sentence((text)) for text in self.data[text_col].values]
        self.labels = self.data[label_col].values

    def encode_sentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ') if word in self.lang.word2index]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_encoded = self.texts[idx]
        label = self.labels[idx]
        #Add EOS token
        text_encoded_with_eos = text_encoded + [EOS_token]
        return torch.tensor(text_encoded_with_eos, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def collate(batch):
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
    batch_sorted = sorted(batch, key= lambda x: len(x[0]), reverse = True)
    texts, labels = zip(*batch_sorted)

    cls_tensor = torch.tensor([CLS_token])
    texts = [torch.cat((cls_tensor, tg)) for tg in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    src_mask = (texts_padded == 0)
    src_mask[:,0] = False

    labels = torch.tensor(labels, dtype=torch.float32)

    return texts_padded, src_mask, labels