import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_PATH = "data/distillation/train_teacher.jsonl"
MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT = "models/actionlens-v0.2"

NUM_LABELS = 6
MAX_LENGTH = 256

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

HARD_WEIGHT = 0.5
TEACHER_WEIGHT = 0.5


class DistillationDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer

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
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),

            "labels": torch.tensor(
                sample["labels"],
                dtype=torch.float,
            ),

            "teacher_labels": torch.tensor(
                sample["teacher_labels"],
                dtype=torch.float,
            ),
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    device = get_device()

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = DistillationDataset(
        DATA_PATH,
        tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"Training samples: {len(dataset)}")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)
            teacher_labels = batch["teacher_labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            # Normal supervised loss
            hard_loss = loss_fn(
                logits,
                labels,
            )

            # Knowledge distillation loss
            teacher_loss = loss_fn(
                logits,
                teacher_labels,
            )

            loss = (
                HARD_WEIGHT * hard_loss
                + TEACHER_WEIGHT * teacher_loss
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Batch {batch_index + 1}/{len(dataloader)} "
                    f"- Loss: {loss.item():.4f} "
                    f"- Hard: {hard_loss.item():.4f} "
                    f"- Teacher: {teacher_loss.item():.4f}"
                )

        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch + 1} finished "
            f"- Average loss: {average_loss:.4f}"
        )

    model.save_pretrained(MODEL_OUTPUT)
    tokenizer.save_pretrained(MODEL_OUTPUT)

    print(
        f"\nDistilled model saved to: {MODEL_OUTPUT}"
    )


if __name__ == "__main__":
    main()