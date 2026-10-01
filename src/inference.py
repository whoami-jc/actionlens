import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_PATH = "models/actionlens-v0.2"
MAX_LENGTH = 256

LABELS = [
    "read_only",
    "mutating",
    "destructive",
    "sensitive_data",
    "external_communication",
    "privileged",
]


class ActionLensClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.device = self._get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        tool: str,
        description: str,
        arguments: dict,
    ) -> dict[str, float]:
        text = self._build_text(
            tool,
            description,
            arguments,
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

        return {
            label: round(probability.item(), 4)
            for label, probability in zip(
                LABELS,
                probabilities,
            )
        }

    @staticmethod
    def _build_text(
        tool: str,
        description: str,
        arguments: dict,
    ) -> str:
        arguments_json = json.dumps(
            arguments,
            ensure_ascii=False,
        )

        return (
            f"Tool: {tool}\n"
            f"Description: {description}\n"
            f"Arguments: {arguments_json}"
        )

    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")