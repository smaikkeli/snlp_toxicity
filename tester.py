import pandas as pd
import loader

# Assuming the Lang class, ToxicityDataset class, and collate function are defined as previously discussed.

# Step 1: Load the dummy dataset to build the vocabulary
df = pd.read_csv('testdata.csv')
lang = loader.Lang("eng")
for sentence in df['text']:
    lang.addSentence(loader.normalizeString(sentence))

# Step 2: Initialize the ToxicityDataset with the dummy dataset
test_dataset = loader.ToxicityDataset('testdata.csv', 'id', 'text', 'label', lang)

# Step 3: Test __getitem__
print("Testing __getitem__:")
for i in range(len(test_dataset)):
    sample = test_dataset[i]
    print(f"Sample {i}: {sample}")

# Step 4: Test DataLoader with collate_fn
from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=loader.collate)



print("\nTesting DataLoader with collate_fn:")
for i, data in enumerate(test_loader):
    inputs, mask, labels = data
    print(f"Batch {i}:")
    print(f"Inputs: {inputs}")
    print(f"Labels: {labels}")


translated_texts, labels = loader.translateDatasetEntries(test_dataset, lang)
for text, label in zip(translated_texts, labels):
    print(f"Text: {text} - Label: {label}")