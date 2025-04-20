import ast
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


import matplotlib
from sklearn.metrics import average_precision_score

matplotlib.use('Agg')  # Должно быть ПЕРЕД импортом plt
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
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
        self.attention_weights = None  # Добавляем атрибут для хранения весов

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs
    ):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=False
        )

        # Сохраняем веса внимания
        attn_output, attn_weights = self.attention(
            outputs.last_hidden_state,
            outputs.last_hidden_state,
            outputs.last_hidden_state,
            key_padding_mask=~attention_mask.bool(),
            average_attn_weights=False
        )
        self.attention_weights = attn_weights.detach().cpu().numpy()

        pooled = torch.mean(attn_output, dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        return (loss, logits) if loss is not None else logits

    def get_attention(self):
        return self.attention_weights


# Dataset class
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
            max_length=299
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
        # Structured input
        text = (
            f"[AGE] {age} [SEX] {sex} "
            f"[EVIDENCES] {evidences} "
            f"[INITIAL] {initial_evidence}"
        )
        return text


class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets,
            weight=self.weights.expand_as(targets)
        )
        return loss


tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
labels_file = 'labels.json'

test_data = pd.read_csv('release_test_patients.csv')

if os.path.exists(labels_file):
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)
else:
    all_labels = sorted({
        disease for diagnoses in test_data['DIFFERENTIAL_DIAGNOSIS']
        for disease, _ in ast.literal_eval(diagnoses)
    })
    with open(labels_file, 'w') as f:
        json.dump(all_labels, f)

label_to_index = {label: idx for idx, label in enumerate(all_labels)}

class_counts = np.zeros(len(all_labels))
for row in test_data['DIFFERENTIAL_DIAGNOSIS']:
    diagnosis = ast.literal_eval(row)
    for disease, prob in diagnosis:
        if disease in label_to_index:
            class_counts[label_to_index[disease]] += 1

# Вычисление весов для каждого класса
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(all_labels) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

test_dataset = MedicalDataset(test_data, tokenizer, label_to_index)
config = ElectraConfig.from_pretrained("./dd_classification_attention_v1/checkpoint-16026")
config.num_labels = len(all_labels)
config.problem_type = "multi_label_classification"

model = CustomElectra.from_pretrained(
    "./dd_classification_attention_v1/checkpoint-16026",
    config=config
).to(device)


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

    mAP = average_precision_score(labels, probs, average='macro')

    return {"DDR": DDR, "DDP": DDP, "DDF1": DDF1, "mAP": mAP}


weighted_loss = WeightedBCELoss(weights=class_weights)

trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    compute_metrics=compute_ddx_metrics,
)
trainer.loss_func = weighted_loss

# results = trainer.evaluate()
# print(results)

# После оценки
sample = test_dataset[1]
with torch.no_grad():
    outputs = model(**{k: v.unsqueeze(0) for k, v in sample.items()})
    attention = model.get_attention()

print(attention)


def visualize_attention(model, sample, tokenizer, layer=0):
    with torch.no_grad():
        outputs = model(**{k: v.unsqueeze(0) for k, v in sample.items()})

    attention = model.get_attention()
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].cpu().numpy())
    text = tokenizer.decode(sample['input_ids'].cpu().numpy())

    # Создаем папку для графиков
    os.makedirs("attention_heads1", exist_ok=True)

    # Определяем типы связей для заголовков
    head_descriptions = {
        0: "Демографические данные (возраст/пол)",
        1: "Основные симптомы",
        2: "Связи симптом-диагноз",
        3: "Хронология развития",
        4: "Сопутствующие признаки",
        5: "Лабораторные показатели",
        6: "Дифференциальная диагностика",
        7: "Контекстные связи"
    }

    for head in range(8):
        plt.figure(figsize=(16, 12))

        # Создаем тепловую карту
        ax = plt.gca()
        im = ax.imshow(attention[layer][head], cmap='viridis')

        # Настраиваем отображение
        plt.title(
            f"Голова {head + 1}: {head_descriptions.get(head, 'Общие связи')}\n"
            f"Текст: {text[:100]}...",
            fontsize=10,
            pad=20
        )
        plt.xticks(
            range(len(tokens)),
            tokens,
            rotation=90,
            fontsize=6
        )
        plt.yticks(
            range(len(tokens)),
            tokens,
            fontsize=6
        )
        plt.colorbar(im, fraction=0.046, pad=0.04)

        # Подписи для ключевых токенов
        key_tokens = ["[AGE]", "[SEX]", "[EVIDENCES]", "[INITIAL]"]
        for i, token in enumerate(tokens):
            if token in key_tokens:
                plt.annotate(
                    token,
                    xy=(i, i),
                    xytext=(-5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='red'
                )

        # Сохраняем в отдельный файл
        plt.savefig(
            f"attention_heads/layer{layer}_head{head}.png",
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()


def analyze_attention_heads(model, dataset, tokenizer, num_samples=10):
    head_descriptions = {
        0: "Демографические данные (возраст/пол)",
        1: "Основные симптомы",
        2: "Связи симптом-диагноз",
        3: "Хронология развития",
        4: "Сопутствующие признаки",
        5: "Лабораторные показатели",
        6: "Дифференциальная диагностика",
        7: "Контекстные связи"
    }

    # Собираем статистику по всем головам
    head_stats = {i: {"tags": defaultdict(float), "pairs": defaultdict(float)}
                  for i in range(8)}

    # Ключевые медицинские токены
    medical_tags = ["[AGE]", "[SEX]", "[EVIDENCES]", "[INITIAL]"]

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        inputs = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        attention = model.get_attention()  # Уже numpy array
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())

        # Анализ для каждой головы
        for head in range(8):
            attn_matrix = attention[0][head]  # Убрать .numpy()

            # Анализ связей с ключевыми токенами
            for i, src_token in enumerate(tokens):
                if src_token in medical_tags:
                    for j, tgt_token in enumerate(tokens):
                        head_stats[head]["tags"][(src_token, tgt_token)] += attn_matrix[i, j]

                # Топ-3 связей для головы
                top_connections = np.argsort(-attn_matrix[i])[:3]
                for j in top_connections:
                    head_stats[head]["pairs"][(tokens[i], tokens[j])] += attn_matrix[i, j]

    # Генерация отчета
    report = []
    for head in range(8):
        # Самые частые связи
        top_pairs = sorted(head_stats[head]["pairs"].items(),
                           key=lambda x: -x[1])[:5]

        # Ключевые медицинские связи
        medical_connections = []
        for (src, tgt), score in head_stats[head]["tags"].items():
            if src in medical_tags and score > 0.1:
                medical_connections.append(f"{src}→{tgt}: {score:.2f}")

        # Формируем описание
        desc = {
            "Номер головы": head + 1,
            "Предполагаемая функция": head_descriptions[head],
            "Топ-5 связей": "\n".join([f"{p[0][0]} → {p[0][1]}: {p[1]:.2f}" for p in top_pairs]),
            "Медицинские связи": "\n".join(medical_connections[:3])
        }
        report.append(desc)

    # Визуализация
    fig, axs = plt.subplots(4, 2, figsize=(20, 25))
    for head in range(8):
        row = head // 2
        col = head % 2

        # Пример матрицы внимания (без .numpy())
        sample_attn = attention[0][head]
        axs[row, col].imshow(sample_attn, cmap='viridis')
        axs[row, col].set_title(
            f"Голова {head + 1}: {head_descriptions[head]}",
            fontsize=10
        )

    plt.savefig("attention_heads_analysis.png")
    plt.close()

    return pd.DataFrame(report)


# Использование после обучения
analysis_report = analyze_attention_heads(model, test_dataset, tokenizer)
analysis_report.to_csv("attention_analysis_report.csv", index=False)
print(analysis_report[["Номер головы", "Предполагаемая функция", "Топ-5 связей"]])