import argparse
import json
from pathlib import Path

from openai import OpenAI


INPUT_PATH = Path("data/processed/train.jsonl")
OUTPUT_PATH = Path("data/distillation/train_teacher.jsonl")

MODEL = "gpt-5"

LABELS = [
    "read_only",
    "mutating",
    "destructive",
    "sensitive_data",
    "external_communication",
    "privileged",
]

client = OpenAI()


def classify_batch(samples):
    items = [
        {
            "id": sample["id"],
            "text": sample["text"],
        }
        for sample in samples
    ]

    prompt = f"""
You are a security classifier for AI agent tool calls.

For every tool call below, assign a probability from 0.0 to 1.0
to each risk label.

Labels:

read_only:
The action only reads or inspects information.

mutating:
The action changes state.

destructive:
The action deletes, removes or irreversibly destroys something.

sensitive_data:
The action accesses or exposes sensitive information.

external_communication:
The action communicates with an external system or person.

privileged:
The action requires or exercises elevated privileges.

Return one classification for every input item.

Tool calls:

{json.dumps(items, ensure_ascii=False)}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "risk_scores_batch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                    },
                                    **{
                                        label: {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 1,
                                        }
                                        for label in LABELS
                                    },
                                },
                                "required": [
                                    "id",
                                    *LABELS,
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
            }
        },
    )

    return json.loads(response.output_text)["results"]


def load_samples(limit):
    samples = []

    with open(INPUT_PATH, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            samples.append(json.loads(line))

            if limit and len(samples) >= limit:
                break

    return samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of samples to label.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples sent to the teacher per request.",
    )

    args = parser.parse_args()

    samples = load_samples(args.limit)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Samples: {len(samples)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Teacher: {MODEL}\n")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output:

        for start in range(
            0,
            len(samples),
            args.batch_size,
        ):
            batch = samples[
                start:start + args.batch_size
            ]

            teacher_results = classify_batch(batch)

            results_by_id = {
                result["id"]: result
                for result in teacher_results
            }

            for sample in batch:
                teacher = results_by_id[sample["id"]]

                sample["teacher_labels"] = [
                    teacher[label]
                    for label in LABELS
                ]

                output.write(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            processed = min(
                start + len(batch),
                len(samples),
            )

            print(
                f"Labeled {processed}/{len(samples)}"
            )

    print(
        f"\nTeacher labels saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()