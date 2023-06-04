import torch
import datasets
import transformers
import numpy as np
from seqeval.metrics import f1_score
import pandas as pd
import os

from train_datasets import BPEDropoutTrainDataset

if torch.cuda.is_available():
    print("GPU is enabled.")
    print("device count: {}, current device: {}".format(torch.cuda.device_count(), torch.cuda.current_device()))
else:
    print("GPU is not enabled.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_dir = "./cache"

dataset_path = "masakhane/masakhaner2"
language = "yor"
# dataset_path = "conll2003"
# language = None

# subword regularization params:
bpe_dropout_p = 0.1
model_path = "xlm-roberta-base"

train_dataset = BPEDropoutTrainDataset(dataset_path, model_path, dataset_language=language, bpe_dropout_p=bpe_dropout_p, cache_dir=cache_dir, train=True)
test_dataset = BPEDropoutTrainDataset(dataset_path, model_path, dataset_language=language, bpe_dropout_p=0.0, cache_dir=cache_dir, train=False)


tags = train_dataset.dset['train'].features["ner_tags"].feature
index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}
tags

tokenizer = train_dataset.tokenizer

training_args = transformers.TrainingArguments(
    output_dir = "./checkpoints/xlm-roberta-ner-yor-2",
    log_level = "error",
    num_train_epochs = 10,
    per_device_train_batch_size = 6,
    per_device_eval_batch_size = 6,
    evaluation_strategy = "epoch",
    fp16 = True,
    logging_steps = len(train_dataset),
    push_to_hub = False
)

def metrics_func(eval_arg):
    preds = np.argmax(eval_arg.predictions, axis=2)
    batch_size, seq_len = preds.shape
    y_true, y_pred = [], []
    for b in range(batch_size):
        true_label, pred_label = [], []
        for s in range(seq_len):
            if eval_arg.label_ids[b, s] != -100:  # -100 must be ignored
                true_label.append(index2tag[eval_arg.label_ids[b][s]])
                pred_label.append(index2tag[preds[b][s]])
        y_true.append(true_label)
        y_pred.append(pred_label)
    return {"f1": f1_score(y_true, y_pred)}

data_collator = transformers.DataCollatorForTokenClassification(
    tokenizer,
    return_tensors="pt")

xlmr_config = transformers.AutoConfig.from_pretrained(
    "xlm-roberta-base",
    num_labels=tags.num_classes,
    id2label=index2tag,
    label2id=tag2index
)

model = (transformers.RobertaForTokenClassification
         .from_pretrained("xlm-roberta-base", config=xlmr_config, cache_dir=cache_dir)
         .to(device))

trainer = transformers.Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    compute_metrics = metrics_func,
    train_dataset = train_dataset,
    eval_dataset = test_dataset
)

trainer.train()
