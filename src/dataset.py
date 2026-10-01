import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ActionLensDataset(Dataset):
    def __init__(
        self,
        path: str,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
    ):
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        with open(path, encoding="utf-8") as file:
            self.samples = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.float,
            ),
        }