import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset import ActionLensDataset
from src.model import create_model


TRAIN_PATH = "data/processed/train.jsonl"
MODEL_OUTPUT = "models/actionlens-v0.1"

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3


def main():
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load dataset
    dataset = ActionLensDataset(TRAIN_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Load model
    model = create_model()
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch_index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_index + 1) % 50 == 0:
                print(
                    f"Batch {batch_index + 1}/{len(dataloader)} "
                    f"- Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch + 1} finished "
            f"- Average loss: {average_loss:.4f}"
        )

    # Save trained model
    model.save_pretrained(MODEL_OUTPUT)

    # Save tokenizer too
    dataset.tokenizer.save_pretrained(MODEL_OUTPUT)

    print(f"\nModel saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
