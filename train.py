import argparse
import os
import shutil
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,confusion_matrix
import math
from get_data import get_datasets
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

parser = argparse.ArgumentParser(description='BERT Classification')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--train_batch_size', default=16, type=int, help='train batchsize')
parser.add_argument('--valid_batch_size', default=256, type=int, help='validation batchsize')
parser.add_argument('--output_path', default="/home/rkxu/workspace/classification/output", type=str,help='save path for output')
parser.add_argument('--log_path', default="/home/rkxu/workspace/classification/log", type=str,help='save path for log')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--max_length', default=128, type=int ,help='sentence max length')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
args = parser.parse_args()

print(args)

def compute_metrics(p):
    preds_list, out_label_list = p.predictions, p.label_ids
    preds_list = np.argmax(preds_list, axis=-1)
    return {
        "accuracy": accuracy_score(preds_list, out_label_list) * 100,
        "f1": f1_score(preds_list, out_label_list) * 100,
        "recall": recall_score(preds_list, out_label_list) * 100,
        "precision": precision_score(preds_list, out_label_list) * 100,
        "confusion_matrix": confusion_matrix(preds_list, out_label_list).tolist()
    }

def train_and_eval(args):
    train_dataset, val_dataset, test_dataset = get_datasets(args)
    warmup_steps = math.ceil(len(train_dataset) / args.train_batch_size * args.epochs * 0.1)
    training_args = TrainingArguments(
        output_dir=args.output_path,  # output directory
        logging_dir=args.log_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy = "epoch",
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.valid_batch_size,  # batch size for evaluation
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        seed=args.seed
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    trainer=Trainer(
        model=model,                    # the instantiated 🤗 Transformers model to be trained
        args=training_args,             # training arguments, defined above
        train_dataset=train_dataset,    # training dataset
        eval_dataset=val_dataset,       # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()


    # trainer.save_model()
    for file in os.listdir(args.output_path):
        if file.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(args.output_path, file))
    result = trainer.evaluate(eval_dataset=test_dataset)
    return result

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_seed(args)
    eval_result = train_and_eval(args)
    for key in eval_result.keys():
        print(f"{key}={eval_result[key]}")
