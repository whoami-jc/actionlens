from src.dataset import ActionLensDataset


dataset = ActionLensDataset(
    "data/processed/train.jsonl"
)

sample = dataset[0]

print(f"Dataset size: {len(dataset)}")
print(f"Input shape: {sample['input_ids'].shape}")
print(f"Mask shape: {sample['attention_mask'].shape}")
print(f"Labels: {sample['labels']}")
