from src.inference import ActionLensClassifier


classifier = ActionLensClassifier()

result = classifier.predict(
    tool="delete_namespace",
    description="Delete a Kubernetes namespace and all its resources",
    arguments={
        "namespace": "production",
    },
)

for label, probability in result.items():
    print(
        f"{label:25} {probability:.4f}"
    )
