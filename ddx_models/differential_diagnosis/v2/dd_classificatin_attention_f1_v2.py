import json
import torch
import pandas as pd
from transformers import AutoTokenizer, Trainer
from differential_diagnosis.v2.dd_classificatin_attention_v2 import CustomElectra, MedicalDataset, compute_ddx_metrics

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    # Загрузка токенизатора и добавление токенов
    tokenizer = AutoTokenizer.from_pretrained("./dd_classification_attention_v2_tokenizer")

    # Загрузка меток
    with open('../../common/labels.json', 'r') as f:
        all_labels = json.load(f)
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    model_path = "./dd_classification_attention_v2_11-12/checkpoint-64102"
    model = CustomElectra.from_pretrained(model_path).to(device)
    # model.resize_token_embeddings(len(tokenizer))  # Синхронизация токенизатора и модели

    # Проверка данных
    test_data = pd.read_csv('release_test_patients.csv')
    print("Пример данных:", test_data.iloc[0]['DIFFERENTIAL_DIAGNOSIS'])

    test_dataset = MedicalDataset(test_data, tokenizer, label_to_index)

    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_ddx_metrics,
    )

    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()

#2 epochs
# {'eval_loss': 0.0565946102142334, 'eval_model_preparation_time': 0.0011, 'eval_DDR': 0.9481489755333883, 'eval_DDP': 0.9194950801035492, 'eval_DDF1': 0.9336022199485412, 'eval_runtime': 650.13, 'eval_samples_per_second': 206.926, 'eval_steps_per_second': 25.867}

#4 epochs
# {'eval_loss': 0.05620526522397995, 'eval_model_preparation_time': 0.0011, 'eval_DDR': 0.9517030228571284, 'eval_DDP': 0.9346772433937448, 'eval_DDF1': 0.9431132988912363, 'eval_runtime': 672.0701, 'eval_samples_per_second': 200.171, 'eval_steps_per_second': 25.023}
