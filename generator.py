import pandas as pd
import unicodedata
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_cosine_schedule_with_warmup


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

def add_labels(row):
        return f"<|startoftext|><|{row['label']}|> {row['text']} <|endoftext|>"

def fine_tune(model_name='gpt2', output_dir='./model_save', num_train_epochs=4, batch_size=2):
    train = pd.read_csv('/content/drive/MyDrive/SNLP/train_2024.csv')
    dev = pd.read_csv('/content/drive/MyDrive/SNLP/dev_2024.csv')

    train['text'] = train['text'].apply(normalizeString)
    train['text'] = train['text'].apply(unicodeToAscii)

    dev['text'] = dev['text'].apply(normalizeString)
    dev['text'] = dev['text'].apply(unicodeToAscii)

    train['text'] = train.apply(add_labels, axis=1)
    dev['text'] = dev.apply(add_labels, axis=1)

    train['text'].to_csv('/content/drive/MyDrive/SNLP/train_dataset.txt', index=False, header=False)
    dev['text'].to_csv('/content/drive/MyDrive/SNLP/dev_dataset.txt', index=False, header=False)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = TextDataset(tokenizer=tokenizer, file_path='train_dataset.txt', block_size=128)
    dev_dataset = TextDataset(tokenizer=tokenizer, file_path='dev_dataset.txt', block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy='steps',
        eval_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

def predict(model, tokenizer, input_text, max_length=160, num_return_sequences=3):
    encoded_prompt = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").input_ids
    encoded_prompt = encoded_prompt.to(model.device) 

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )

    generated_sequences = []
    for generated_sequence in output_sequences:
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        generated_sequences.append(text.strip())

    return generated_sequences

model, tokenizer = fine_tune()

input_text = "<|startoftext|><|1|> <|startoftext|><|1|>"
generated_texts = predict(model, tokenizer, input_text)
for text in generated_texts:
    print(text)