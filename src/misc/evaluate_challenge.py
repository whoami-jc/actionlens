import json

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "models/actionlens-v0.2"
CHALLENGE_PATH = "data/challenge/unseen.jsonl"

THRESHOLD = 0.5
MAX_LENGTH = 256
ERRORS_TO_SHOW = 5

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


def build_text(sample):
    arguments = json.dumps(
        sample["arguments"],
        ensure_ascii=False,
    )

    return (
        f"Tool: {sample['tool']}\n"
        f"Description: {sample['description']}\n"
        f"Arguments: {arguments}"
    )


def load_challenge():
    samples = []

    with open(CHALLENGE_PATH, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def print_errors(errors, title):
    print(f"\n{title}")
    print("=" * len(title))

    if not errors:
        print("None")
        return

    for error in errors[:ERRORS_TO_SHOW]:
        print(f"\nLabel:       {error['label']}")
        print(f"Tool:        {error['tool']}")
        print(f"Description: {error['description']}")
        print(f"Probability: {error['probability']:.4f}")
        print(f"Expected:    {error['expected']}")


def main():
    device = get_device()

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    samples = load_challenge()

    predictions = []
    true_labels = []

    false_negatives = []
    false_positives = []

    for sample in samples:
        text = build_text(sample)

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        probabilities = torch.sigmoid(
            outputs.logits
        )[0].cpu()

        predicted = (
            probabilities >= THRESHOLD
        ).int().numpy()

        expected = [
            int(sample["labels"][label])
            for label in LABELS
        ]

        predictions.append(predicted)
        true_labels.append(expected)

        for index, label in enumerate(LABELS):
            probability = probabilities[index].item()

            error = {
                "label": label,
                "tool": sample["tool"],
                "description": sample["description"],
                "probability": probability,
                "expected": expected[index],
            }

            if expected[index] == 1 and predicted[index] == 0:
                false_negatives.append(error)

            elif expected[index] == 0 and predicted[index] == 1:
                false_positives.append(error)

    print(f"\nChallenge samples: {len(samples)}")
    print("\nActionLens Unseen Challenge\n")

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # Most confident mistakes:
    #
    # FN: lowest probability despite expected=1
    # FP: highest probability despite expected=0

    false_negatives.sort(
        key=lambda error: error["probability"]
    )

    false_positives.sort(
        key=lambda error: error["probability"],
        reverse=True,
    )

    print_errors(
        false_negatives,
        "Most confident false negatives",
    )

    print_errors(
        false_positives,
        "Most confident false positives",
    )


if __name__ == "__main__":
    main()