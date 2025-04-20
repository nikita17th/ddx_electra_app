# continue_training.py
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer
from differential_diagnosis.v2.dd_classificatin_attention_v2 import CustomElectra, MedicalDataset, compute_ddx_metrics


# Убедитесь, что классы и функции из исходного кода доступны

def main():
    # Загрузка токенизатора и добавление токенов
    tokenizer = AutoTokenizer.from_pretrained("./dd_classification_attention_v2_tokenizer")

    # Загрузка меток
    with open('labels.json', 'r') as f:
        all_labels = json.load(f)
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    model_path = "./dd_classification_attention_v2_9-10/checkpoint-64102"  # Путь к сохраненной модели
    config = CustomElectra.config_class.from_pretrained(model_path)
    model = CustomElectra(config).from_pretrained(model_path)
    model.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    # Загрузка новых данных для дообучения
    new_train_data = pd.read_csv('release_train_patients.csv')  # Путь к новым данным
    new_val_data = pd.read_csv('release_validate_patients.csv')      # Валидационные данные

    # Подготовка датасетов
    train_dataset = MedicalDataset(new_train_data, tokenizer, label_to_index)
    val_dataset = MedicalDataset(new_val_data, tokenizer, label_to_index)

    training_args = TrainingArguments(
        output_dir="./dd_classification_attention_v2_11-12",  # Новый выходной каталог
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        num_train_epochs=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir="./logs_continued",
        fp16=False,
        dataloader_pin_memory=False
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_ddx_metrics,
    )

    # Запуск дообучения
    trainer.train()

if __name__ == "__main__":
    main()