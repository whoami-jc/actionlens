import argparse
import json
import random
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

LABELS = [
    "read_only",
    "mutating",
    "destructive",
    "sensitive_data",
    "external_communication",
    "privileged",
]


def load_data():
    samples = []

    # Join all the files in a single one
    for file in RAW_DIR.glob("*.jsonl"):
        with open(file, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

    return samples


def prepare_sample(sample):
    text = (
        f"Tool: {sample['tool']}\n"
        f"Description: {sample['description']}\n"
        f"Arguments: {json.dumps(sample['arguments'], ensure_ascii=False)}"
    )

    # keep the labels in the same order as LABELS
    labels = [
        int(sample["labels"][label])
        for label in LABELS
    ]

    return {
        "id": sample["id"],
        "family": sample["family"],
        "text": text,
        "labels": labels,
    }


def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)

    args = parser.parse_args()

    if round(args.train + args.validation + args.test, 10) != 1.0:
        raise ValueError("Train, validation and test percentages must add up to 1.")

    samples = [prepare_sample(s) for s in load_data()]

    # Fixed seed for reproducible shuffling with the same input
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)

    train_end = int(n * args.train)
    validation_end = train_end + int(n * args.validation)

    # Split the samples into train, validation, and test sets
    train = samples[:train_end]
    validation = samples[train_end:validation_end]
    test = samples[validation_end:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(OUTPUT_DIR / "train.jsonl", train)
    save_jsonl(OUTPUT_DIR / "validation.jsonl", validation)
    save_jsonl(OUTPUT_DIR / "test.jsonl", test)

    print(f"Total:      {n}")
    print(f"Train:      {len(train)} ({args.train:.0%})")
    print(f"Validation: {len(validation)} ({args.validation:.0%})")
    print(f"Test:       {len(test)} ({args.test:.0%})")


if __name__ == "__main__":
    main()
