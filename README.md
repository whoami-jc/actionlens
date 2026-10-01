# Actionlens 

AI Agent Tool Call Risk Classifier

## Table of contents

- [Purpose](#purpose)
- [Dataset](#dataset)
- [Tests](#tests)

## Purpose

## Dataset

The dataset contains **8,000 real tool-call examples**, organized into eight categories under `data/raw/`. Each category has its own JSONL file, with one example per line.

| Category | File | Examples |
|---|---|---:|
| Cloud | `cloud.jsonl` | 1,000 |
| Database | `database.jsonl` | 1,000 |
| Email | `email.jsonl` | 1,000 |
| Filesystem | `filesystem.jsonl` | 1,000 |
| Git | `git.jsonl` | 1,000 |
| HTTP | `http.jsonl` | 1,000 |
| Mixed | `mixed.jsonl` | 1,000 |
| Shell | `shell.jsonl` | 1,000 |
| **Total** | | **8,000** |

### Preparing the dataset

[`data/prepare_dataset.py`](data/prepare_dataset.py) combines the raw files, formats each tool's name, description, and arguments into a single text input, and converts its labels into a binary vector in this order: `read_only`, `mutating`, `destructive`, `sensitive_data`, `external_communication`, `privileged`. Each processed example also retains its `id` and `family`.

The script shuffles the examples with a fixed seed (`42`) and splits them into training, validation, and test sets. Run it from the project root:

```bash
poetry run python data/prepare_dataset.py
```

The default split is **80% training, 10% validation, and 10% test** (6,400 / 800 / 800 examples). To choose different proportions:

```bash
poetry run python data/prepare_dataset.py --train 0.7 --validation 0.15 --test 0.15
```
## Tests

TODO