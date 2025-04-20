import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    ElectraForSequenceClassification,
    Trainer,
    TrainingArguments,
    ElectraConfig
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CustomElectra(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Механизм внимания с 4 головами
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_labels)
        )
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )

        # Применение механизма внимания
        attn_output, _ = self.attention(
            outputs.last_hidden_state,
            outputs.last_hidden_state,
            outputs.last_hidden_state,
            key_padding_mask=~attention_mask.bool()
        )

        # Усреднение с учетом внимания
        pooled = torch.mean(attn_output, dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        return (loss, logits) if loss is not None else logits


class MedicalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_to_index):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = self._preprocess_row(row)

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=300
        )

        labels = torch.zeros(len(self.label_to_index))
        diagnosis = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])

        for disease, probability in diagnosis:
            if disease in self.label_to_index:
                index = self.label_to_index[disease]
                labels[index] = probability

        return {
            'input_ids': inputs['input_ids'].squeeze(0).to(device),
            'attention_mask': inputs['attention_mask'].squeeze(0).to(device),
            'labels': labels.to(device)
        }

    @staticmethod
    def _preprocess_row(row):
        age = str(row['AGE'])
        sex = row['SEX']
        evidences = ' '.join(ast.literal_eval(row['EVIDENCES']))
        initial_evidence = row['INITIAL_EVIDENCE']
        return f"[AGE] {age} [SEX] {sex} [EVIDENCES] {evidences} [INITIAL] {initial_evidence}"


def load_evidences_codes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_codes = []
    seen = set()

    for evidence_id, evidence_data in data.items():
        if evidence_id not in seen:
            all_codes.append(evidence_id)
            seen.add(evidence_id)

        possible_values = evidence_data.get("possible-values", [])
        for val in possible_values:
            code = f"{evidence_id}_@_{val}"
            if code not in seen:
                all_codes.append(code)
                seen.add(code)

    return all_codes


def compute_ddx_metrics(eval_pred):
    logits, labels = eval_pred

    # Преобразуем логиты в вероятности
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid через numpy
    predictions = probs > 0.02  # Порог остается 0.02

    recall_values = []
    precision_values = []

    for true, pred in zip(labels, predictions):
        true_indices = set(np.where(true > 0)[0].tolist())
        pred_indices = set(np.where(pred)[0].tolist())

        # Recall (DDR)
        if len(true_indices) > 0:
            recall = len(true_indices & pred_indices) / len(true_indices)
        else:
            recall = 0
        recall_values.append(recall)

        # Precision (DDP)
        if len(pred_indices) > 0:
            precision = len(true_indices & pred_indices) / len(pred_indices)
        else:
            precision = 0
        precision_values.append(precision)

    DDR = np.mean(recall_values)
    DDP = np.mean(precision_values)
    DDF1 = 2 * DDR * DDP / (DDR + DDP) if (DDR + DDP) > 0 else 0

    labels_binary = (labels >= 0.05).astype(int)
    valid_classes = np.where(labels_binary.sum(axis=0) > 0)[0]
    if len(valid_classes) == 0:
        mAP = 0.0  # Все классы "пустые"
    else:
        mAP = average_precision_score(
            labels_binary[:, valid_classes],
            probs[:, valid_classes],
            average='macro'
        )
    return {"DDR": DDR, "DDP": DDP, "DDF1": DDF1, "mAP": mAP}


def main():
    # Загрузка и добавление уникальных токенов
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    evidences_codes = load_evidences_codes('release_evidences.json')
    print(evidences_codes)
    tokenizer.add_tokens(evidences_codes)

    train_data = pd.read_csv('release_train_patients.csv')
    val_data = pd.read_csv('release_validate_patients.csv')

    # Обработка меток
    labels_file = 'labels.json'
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            all_labels = json.load(f)
    else:
        all_labels = sorted({
            disease for diagnoses in train_data['DIFFERENTIAL_DIAGNOSIS']
            for disease, _ in ast.literal_eval(diagnoses)
        })
        with open(labels_file, 'w') as f:
            json.dump(all_labels, f)

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Инициализация модели
    config = ElectraConfig.from_pretrained("google/electra-small-discriminator")
    config.num_labels = len(all_labels)
    config.problem_type = "multi_label_classification"
    model = CustomElectra(config)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)  # Важно после добавления токенов!
    model.to(device)

    # Подготовка данных
    train_dataset = MedicalDataset(train_data, tokenizer, label_to_index)
    val_dataset = MedicalDataset(val_data, tokenizer, label_to_index)

    # Конфигурация обучения
    torch.mps.empty_cache()
    training_args = TrainingArguments(
        output_dir="./dd_classification_attention_v2",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        num_train_epochs=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir="./logs",
        fp16=False,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_ddx_metrics,
    )
    tokenizer.save_pretrained("./dd_classification_attention_v2_tokenizer")

    # Запуск обучения
    trainer.train()
    tokenizer.save_pretrained("./dd_classification_attention_v2_tokenizer_after_train")


if __name__ == "__main__":
    main()
