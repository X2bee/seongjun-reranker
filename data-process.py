from datasets import load_dataset
from huggingface_hub import login
login("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")

# 데이터셋 로드
ds = load_dataset("williamjeong2/msmarco-triplets-ko-v1")

# 데이터셋 일부 출력 함수
def preview_dataset(dataset, num_samples=5):
	print(f"Dataset preview ({num_samples} samples):")
	for i in range(num_samples):
		print(f"Sample {i + 1}:")
		print(dataset[i])
		print("-" * 50)

# train 데이터셋 확인 (또는 다른 스플릿 사용 가능)
if "train" in ds:
	preview_dataset(ds["train"], num_samples=5)
else:
	print("No 'train' split found in the dataset.")
