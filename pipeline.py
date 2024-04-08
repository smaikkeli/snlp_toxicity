
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

from loader import Lang, ToxicityDataset, normalizeString, collate

print(f'Loading data... \n')

lang = Lang("eng")
data = pd.read_csv('data/train_2024.csv', quoting = 3)
df = pd.DataFrame(data)
for sentence in df['text']:
    lang.addSentence(normalizeString(sentence))   

trainset = ToxicityDataset('data/train_2024.csv', 'id', 'text', 'label', lang)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)

#Test if trainset contains correct number of rows
if len(trainset) == 99000:
    print("Trainset loaded correctly")

print("Trainset loaded \n")

from encoder import Encoder, MLPClassifier, EncoderClassifier

embed_size = 256

#This correponds to the first model I tried
bert_encoder = Encoder(src_vocab_size=trainset.lang.n_words, n_blocks = 3, n_features = embed_size, n_heads = 4, n_hidden = 512, dropout = 0.1, max_length = 5000)
classifier = MLPClassifier(n_features = embed_size, num_classes = 2, num_layers = 2, dropout = 0.1)
encoder_classifier = EncoderClassifier(bert_encoder, classifier)

print("Model loaded \n")

model = encoder_classifier
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas = (0.9, 0.98), eps=1e-9)
criterion = nn.BCEWithLogitsLoss()

print(f'Entering the training loop \n')

epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total_accuracy = 0.0
    
    # Set the model to training mode
    model.train()
    
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        inputs, mask, labels = data
    
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(torch.float32).reshape(labels.size(0), 1).to(device)
        
        outputs = model(inputs, mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # Compute total loss
        total_loss += loss.item()

        # Calculate accuracy
        predicted = torch.round(torch.sigmoid(outputs))
        correct = (predicted == labels).sum().item()
        total = labels.size(0)

        # Print batch loss
        #if i % 100 == 0:
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item():.4f}")

        total_loss += total_loss / len(train_loader)
        total_accuracy += correct / total * 100.0

    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {total_accuracy:.2f}%")

# Save the model

torch.save(model.state_dict(), 'encoder.pth')

print(f'Model saved \n')

#If model is not defined, load it from encoder.pth
bert_encoder = Encoder(src_vocab_size=trainset.lang.n_words, n_blocks = 3, n_features = embed_size, n_heads = 4, n_hidden = 512, dropout = 0.1, max_length = 5000)
classifier = MLPClassifier(n_features = embed_size, num_classes = 2, num_layers = 2, dropout = 0.1)
model = EncoderClassifier(bert_encoder, classifier)
model.load_state_dict(torch.load('encoder.pth'))

print("Model loaded correctly \n")

print('Fetching the test data \n')

test_data = pd.read_csv('./data/test_2024.csv', quoting = 3)

test_data['label'] = -1

test_df = pd.DataFrame(test_data)
testset = ToxicityDataset('./test_data.csv', 'id', 'text', 'label', lang)
test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate)

#Test if testset contains correct number of rows)
predictions = pd.DataFrame(columns = ['id', 'label'])
model.eval()

print(f'Predicting labels... \n')

predicted = torch.tensor([]).to(device)
translations = []
for i, data in enumerate(test_loader):
    inputs, mask, labels = data
    
    #translated = [testset.lang.index2word[i.item()] for input in inputs for i in input]
    #translations.append(translated)
    inputs = inputs.to(device)
    mask = mask.to(device)
    
    outputs = model(inputs, mask)

    pre = torch.round(torch.sigmoid(outputs))
    predicted = torch.cat((predicted, pre), dim = 0)
    #if i % 200 == 0:
        #print(i)


predicted = predicted.squeeze().detach().numpy().astype(int)
predictions['label'] = predicted
predictions['id'] = test_data['id']

print(f'Predictions done')

predictions.to_csv('predictions.csv', index = False, header = True)