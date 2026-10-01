import torch

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from src.dataset import ActionLensDataset


MODEL_PATH = "models/actionlens-v0.1"
TEST_PATH = "data/processed/test.jsonl"

BATCH_SIZE = 32
THRESHOLD = 0.5

LABELS = [
    "read_only",
    "mutating",
    "destructive",
    "sensitive_data",
    "external_communication",
    "privileged",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    device = get_device()

    print(f"Using device: {device}")

    dataset = ActionLensDataset(
        TEST_PATH,
        model_name=MODEL_PATH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.sigmoid(outputs.logits)

            predicted_labels = (
                probabilities >= THRESHOLD
            ).int()

            predictions.extend(
                predicted_labels.cpu().numpy()
            )

            true_labels.extend(
                batch["labels"].int().numpy()
            )

    print("\nActionLens Evaluation\n")

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
