import ast
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments


class MedicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_to_index):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        age = str(row['AGE'])
        sex = row['SEX']
        evidences = ' '.join(ast.literal_eval(row['EVIDENCES']))
        initial_evidence = row['INITIAL_EVIDENCE']
        text = f"Age: {age}, Sex: {sex}. Evidences: {evidences} Initial evidence: {initial_evidence}."

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=299           # change from 128
        )

        labels = torch.zeros(len(self.label_to_index))
        diagnosis = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])

        for disease, probability in diagnosis:
            if disease in self.label_to_index:
                index = self.label_to_index[disease]
                labels[index] = probability

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels
        }


tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
train_data = pd.read_csv('release_train_patients.csv')
labels_file = 'labels.json'

if os.path.exists(labels_file):
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)
else:
    all_labels = sorted(
        {disease for diagnoses in train_data['DIFFERENTIAL_DIAGNOSIS'] for disease, _ in ast.literal_eval(diagnoses)})
    with open(labels_file, 'w') as f:
        json.dump(all_labels, f)

label_to_index = {label: idx for idx, label in enumerate(all_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

train_dataset = MedicalDataset(train_data, tokenizer, label_to_index)

model = ElectraForSequenceClassification.from_pretrained(
    "google/electra-small-discriminator",
    num_labels=len(all_labels),
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir="./dd_classification_v4",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy='steps',
    eval_steps=20000
)

val_data = pd.read_csv('release_validate_patients.csv')
val_dataset = MedicalDataset(val_data, tokenizer, label_to_index)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


# Средняя длина: 142.21578156048838
# Максимальная длина: 299
# 95-й перцентиль: 239.0
# 90-й перцентиль: 218.0