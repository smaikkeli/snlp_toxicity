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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

from torch.nn.utils.rnn import pad_sequence
from loader import Lang, ToxicityDataset, normalizeString, collate

lang = Lang("eng")
data = pd.read_csv('data/train_2024.csv')
df = pd.DataFrame(data)
for sentence in df['text']:
    lang.addSentence(normalizeString(sentence))   

trainset = ToxicityDataset('data/train_2024.csv', 'id', 'text', 'label', lang)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=collate)

#Get random sample from trainset
src_seq, label = trainset[np.random.choice(len(trainset))]
print('Source sentence:')
print(' as word indices: ', src_seq)
print(' as string: ', ' '.join(trainset.lang.index2word[i.item()] for i in src_seq))

test_data = pd.read_csv('data/test_2024.csv')
test_df = pd.DataFrame(test_data)
test_lang = Lang("eng")
for sentence in test_df['text']:
    test_lang.addSentence(normalizeString(sentence))

testset = ToxicityDataset('data/test_2024.csv', 'id', 'text', 'label', test_lang)
test_loader = DataLoader(testset, batch_size=len(testset), shuffle=True, collate_fn=collate)

class PositionalEncoding(nn.Module):
    """This implementation is the same as in the Annotated transformer blog post
        See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """
    def __init__(self, d_model, dropout=0.1, max_len=300):
        assert (d_model % 2) == 0, 'd_model should be an even number.'
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
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
            nn.Linear(n_features, n_features * 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        #Append all layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(n_features * 4, n_features * 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        #The output layer
        if num_classes == 2:
            layers.append(nn.Linear(n_features * 4, 1))
        else:
            layers.append(nn.Linear(n_features * 4, num_classes))
        
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
        
        # Take the [CLS] token
        # x = x[:, 0, :]
        x = x.mean(dim=1)
        
        x = self.classifier(x)

        # Apply sigmoid activation for binary classification
        #x = torch.sigmoid(x)
        
        return x
    
class SimpleEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(SimpleEncoderClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Final classification layer with a single output unit and sigmoid activation
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        # Take the mean over the sequence dimension
        x = x.mean(dim=1)

        # Pass through classifier
        x = self.classifier(x)

        return x

    
embed_size = 256

#This correponds to the first model I tried
bert_encoder = Encoder(src_vocab_size=trainset.lang.n_words, n_blocks = 3, n_features = embed_size, n_heads = 4, n_hidden = 512, dropout = 0.1, max_length = 5000)
simple_encoder = Encoder(src_vocab_size = trainset.lang.n_words, n_blocks = 2, n_features = embed_size, n_heads = 1, n_hidden = 64, dropout = 0.1, max_length = 5000)
classifier = MLPClassifier(n_features = embed_size, num_classes = 2, num_layers = 2, dropout = 0.1)
encoder_classifier = EncoderClassifier(bert_encoder, classifier)

#This is a simpler one with much less parameters
simple_encoder = Encoder(src_vocab_size = trainset.lang.n_words, n_blocks = 2, n_features = 128, n_heads = 4, n_hidden = 256, dropout = 0.2, max_length = 5000)
simple_classifier = MLPClassifier(n_features = 128, num_classes = 2, num_layers = 1, dropout = 0.2)
encoder_classifier_2 = EncoderClassifier(simple_encoder, simple_classifier)

#This is the simplest one, which implements pytorch builtin encoder layers and a simple classifier in one class
simple_encoder_classifier = SimpleEncoderClassifier(vocab_size = trainset.lang.n_words, embedding_dim=256)

model = encoder_classifier
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas = (0.9, 0.98), eps=1e-9)

#Calculate the positive weight fraction
positive  = sum([label for _, label in trainset])
negative = len(trainset) - positive
positive_weight = negative/positive
#criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(positive_weight).to(device))
criterion = nn.BCEWithLogitsLoss()

epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Set the model to training mode
    model.train()
    
    for i, data in enumerate(train_loader):
        inputs, mask, labels = data

        inputs = inputs.to(device)
        mask = mask.to(torch.float32).to(device)
        labels = labels.to(torch.float32).reshape(labels.size(0), 1).to(device)

        optimizer.zero_grad()

        outputs = model(inputs, mask)
        loss = criterion(outputs, labels)

        # Print the outputs, labels, and loss for debugging
        #print(f"Outputs: {outputs}")
        #print(f"Labels: {labels}")
        #print(f"Loss: {loss}")

        loss.backward()
        optimizer.step()

        # Compute total loss
        total_loss += loss.item()

        # Calculate accuracy
        predicted = torch.round(torch.sigmoid(outputs))
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Print batch loss
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")

# Calculate epoch-level statistics
epoch_loss = total_loss / len(train_loader)
epoch_accuracy = correct / total * 100.0

print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        inputs, mask, labels = data
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        outputs = model(inputs, mask)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total