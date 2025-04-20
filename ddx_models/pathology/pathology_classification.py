import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments


class MedicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        age = str(row['AGE'])
        sex = row['SEX']
        evidences = ' '.join(eval(row['EVIDENCES']))
        initial_evidence = row['INITIAL_EVIDENCE']
        text = f"Age: {age}, Sex: {sex}. Evidences: {evidences} Initial evidence: {initial_evidence}."

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=128
        )

        label = row['PATHOLOGY']
        label_id = label_to_id[label]
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id)
        }


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google/electra-small-discriminator")

train_data = pd.read_csv('release_train_patients.csv')
train_dataset = MedicalDataset(train_data, tokenizer)

num_labels = len(train_data['PATHOLOGY'].unique())
unique_labels = train_data['PATHOLOGY'].unique()
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

model = ElectraForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="google/electra-small-discriminator",
    num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./pathology_classification",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    logging_dir="./logs",
)

val_data = pd.read_csv('release_validate_patients.csv')
val_dataset = MedicalDataset(val_data, tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
