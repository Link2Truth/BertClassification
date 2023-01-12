import numpy as np
import pandas as pd 
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures
from datasets import load_dataset
from torch.utils.data import Dataset

def get_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_data = load_dataset("SetFit/sst2",split='train')
    valid_data = load_dataset("SetFit/sst2",split='validation')
    test_data = load_dataset("SetFit/sst2",split='test')

    train_texts = list(train_data['text'])     
    valid_texts = list(valid_data['text'])
    test_texts = list(test_data['text'])

    train_labels = list(train_data['label'])
    valid_labels = list(valid_data['label'])
    test_labels = list(test_data['label'])

    print(f"train:{len(train_texts)},val:{len(valid_texts)},test:{len(test_texts)}")

    train_dataset = general_dataset(train_texts,train_labels,tokenizer, args)
    valid_dataset = general_dataset(valid_texts,valid_labels,tokenizer, args)
    test_dataset = general_dataset(test_texts,test_labels,tokenizer, args)
    return train_dataset, valid_dataset, test_dataset

class general_dataset(Dataset):
    def __init__(self, text, label, tokenizer, args):
        super(Dataset, self).__init__()
        self.texts = [e for e in text]
        self.labels = [e for e in label]
        self.features = []
        tmp_features = tokenizer(self.texts, max_length=args.max_length, padding='max_length', truncation=True)
        for i in range(len(self.texts)):
            self.features.append(InputFeatures(
                input_ids=tmp_features['input_ids'][i], 
                attention_mask=tmp_features['attention_mask'][i],
                token_type_ids=tmp_features['token_type_ids'][i], 
                label=self.labels[i]
            ))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]