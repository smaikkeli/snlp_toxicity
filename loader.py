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
    
    def trim(self):
        """Remove words below a certain frequency threshold"""
        keep_words = [k for k, v in self.word2count.items() if v >= 5]
        print(f'Keeping {len(keep_words)} words out of {len(self.word2count)}')
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
        self.data = pd.read_csv(filename)
        self.lang = lang
        self.texts = [self.encode_sentence(normalizeString(text)) for text in self.data[text_col].values]
        self.labels = self.data[label_col].values

    def encode_sentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ') if word in self.lang.word2index]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_encoded = self.texts[idx]
        label = self.labels[idx]
        #Add EOS token
        text_encoded.append(EOS_token)
        return torch.tensor(text_encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)

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

    labels = torch.tensor(labels, dtype=torch.float)

    return texts_padded, src_mask, labels


'''
lang = Lang("eng")

data = pd.read_csv('data/train_2024.csv')

df = pd.DataFrame(data)

for sentence in df['text']:
    lang.addSentence(normalizeString(sentence))   

train_dataset = ToxicityDataset('data/train_2024.csv', 'id', 'text', 'label', lang)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate)

for texts, src_mask, labels in train_loader:
    print(texts)
    print(src_mask)
    print(labels)
    break
'''