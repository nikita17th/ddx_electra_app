import ast
import json

import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
            max_length=299
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
labels_file = 'labels.json'

with open(labels_file, 'r') as f:
    all_labels = json.load(f)

label_to_index = {label: idx for idx, label in enumerate(all_labels)}

model = ElectraForSequenceClassification.from_pretrained("./dd_classification_v8/checkpoint-32052")
test_data = pd.read_csv('release_test_patients.csv')
test_dataset = MedicalDataset(test_data, tokenizer, label_to_index)


def compute_ddx_metrics(eval_pred):
    logits, labels = eval_pred

    labels = torch.tensor(labels)
    logits = torch.tensor(logits)

    predictions = torch.sigmoid(logits) > 0.02

    # DDR и DDP
    recall_values = []
    precision_values = []

    for true, pred in zip(labels, predictions):
        true_set = set(torch.where(true > 0)[0].tolist())
        pred_set = set(torch.where(pred > 0)[0].tolist())

        # Полнота (Recall)
        if len(true_set) > 0:
            recall = len(true_set & pred_set) / len(true_set)
        else:
            recall = 0
        recall_values.append(recall)

        # Точность (Precision)
        if len(pred_set) > 0:
            precision = len(true_set & pred_set) / len(pred_set)
        else:
            precision = 0
        precision_values.append(precision)

    DDR = sum(recall_values) / len(recall_values)
    DDP = sum(precision_values) / len(precision_values)

    # DDF1
    if DDR + DDP > 0:
        DDF1 = 2 * DDR * DDP / (DDR + DDP)
    else:
        DDF1 = 0

    return {
        'DDR': DDR,
        'DDP': DDP,
        'DDF1': DDF1
    }


trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    compute_metrics=compute_ddx_metrics,
)

results = trainer.evaluate()
print(results)

# predictions = torch.sigmoid(logits) > 0.01
# 100%|██████████| 16817/16817 [07:03<00:00, 39.68it/s]
# {'eval_loss': 0.0584070086479187, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9830555802810353, 'eval_DDP': 0.8183681936532216, 'eval_DDF1': 0.8931839705193885, 'eval_runtime': 424.0301, 'eval_samples_per_second': 317.263, 'eval_steps_per_second': 39.66}

# predictions = torch.sigmoid(logits) > 0.005
# 100%|██████████| 16817/16817 [07:05<00:00, 39.51it/s]
# {'eval_loss': 0.0584070086479187, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9927572157220995, 'eval_DDP': 0.7741807046287505, 'eval_DDF1': 0.8699496138952074, 'eval_runtime': 425.8421, 'eval_samples_per_second': 315.913, 'eval_steps_per_second': 39.491}


# predictions = torch.sigmoid(logits) > 0.01
# 2 epochs max_length=299
# 100%|██████████| 16817/16817 [09:42<00:00, 28.85it/s]
# {'eval_loss': 0.05650928243994713, 'eval_model_preparation_time': 0.0013, 'eval_DDR': 0.9828111395343275, 'eval_DDP': 0.8744105755070251, 'eval_DDF1': 0.9254473466198856, 'eval_runtime': 583.0806, 'eval_samples_per_second': 230.721, 'eval_steps_per_second': 28.842}

# 5 epochs max_length=299
# 100%|██████████| 16817/16817 [09:30<00:00, 29.50it/s]
# {'eval_loss': 0.055520765483379364, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9885332278842606, 'eval_DDP': 0.9250244624709385, 'eval_DDF1': 0.9557249539610834, 'eval_runtime': 570.2423, 'eval_samples_per_second': 235.916, 'eval_steps_per_second': 29.491}

# 7 epochs max_length=299
# 100%|██████████| 16817/16817 [09:38<00:00, 29.06it/s]
# {'eval_loss': 0.05534140765666962, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9896959018698657, 'eval_DDP': 0.9347674736890367, 'eval_DDF1': 0.9614477985507096, 'eval_runtime': 578.9293, 'eval_samples_per_second': 232.376, 'eval_steps_per_second': 29.048}

# 9  epochs max_length=299
# 00%|██████████| 16817/16817 [09:37<00:00, 29.13it/s]
# {'eval_loss': 0.05524466186761856, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9904951529971612, 'eval_DDP': 0.9403233102436205, 'eval_DDF1': 0.9647573801249735, 'eval_runtime': 577.6289, 'eval_samples_per_second': 232.899, 'eval_steps_per_second': 29.114}
#

# 11  epochs max_length=299
# 100%|██████████| 16817/16817 [1:03:09<00:00,  4.44it/s]
# {'eval_loss': 0.05523603782057762, 'eval_model_preparation_time': 0.0012, 'eval_DDR': 0.9718151940728957, 'eval_DDP': 0.9643527726625796, 'eval_DDF1': 0.968069602452894, 'eval_runtime': 3789.6303, 'eval_samples_per_second': 35.499, 'eval_steps_per_second': 4.438}
#
# Process finished with exit code 0
