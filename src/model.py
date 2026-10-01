from transformers import AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "read_only",
    "mutating",
    "destructive",
    "sensitive_data",
    "external_communication",
    "privileged",
]

ID2LABEL = {
    index: label
    for index, label in enumerate(LABELS)
}

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}


def create_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )