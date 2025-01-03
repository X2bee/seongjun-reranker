import argparse
import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	Trainer,
	TrainingArguments,
	DataCollatorWithPadding,
	set_seed
)

def parse_args():
	parser = argparse.ArgumentParser(description="Train a reranker model on Korean MS MARCO dataset")
	parser.add_argument("--hf_token", type=str, default="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")
	parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-reranker-large")
	parser.add_argument("--output_dir", type=str, default="./model_output")
	parser.add_argument("--learning_rate", type=float, default=5e-6)
	parser.add_argument("--fp16", action="store_true")
	parser.add_argument("--num_train_epochs", type=int, default=1)
	parser.add_argument("--per_device_train_batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=50)
	parser.add_argument("--save_total_limit", type=int, default=1)
	parser.add_argument("--max_len", type=int, default=512)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_samples", type=int, default=100)
	return parser.parse_args()

def setup_environment():
	# MPS 관련 설정 비활성화
	os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
	if hasattr(torch.backends, 'mps'):
		torch.backends.mps.enabled = False

def get_device():
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
	else:
		device = torch.device("cpu")
		print("Using CPU")
	return device

def load_tokenizer_and_model(args, device):
	print(f"\nLoading tokenizer and model from {args.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
	model.to(device)
	return tokenizer, model

def prepare_dataset(args):
	print("\nLoading and preparing dataset...")
	ds = load_dataset("williamjeong2/msmarco-triplets-ko-v1")
	train_ds = ds["train"].select(range(args.num_samples))
	print(f"Training on {len(train_ds)} samples")
	return train_ds

def create_preprocess_function(tokenizer, args):
	def preprocess_function(examples):
		queries = [str(q) for q in examples["query"]]
		positives = [str(p) for p in examples["pos"]]
		negatives = [str(n) for n in examples["neg"]]

		all_pairs = []
		all_labels = []

		for q, p, n in zip(queries, positives, negatives):
			all_pairs.extend([[q, p], [q, n]])
			all_labels.extend([1, 0])

		encoded = tokenizer(
			text=all_pairs,
			padding=False,
			truncation=True,
			max_length=args.max_len,
			return_tensors=None,
			is_split_into_words=False
		)

		# Convert labels to float32 to avoid type mismatch
		return {
			"input_ids": encoded["input_ids"],
			"attention_mask": encoded["attention_mask"],
			"labels": torch.tensor(all_labels, dtype=torch.float32)
		}
	return preprocess_function

def get_training_args(args, device):
	return TrainingArguments(
		output_dir=args.output_dir,
		overwrite_output_dir=True,
		learning_rate=args.learning_rate,
		fp16=args.fp16 if device.type == "cuda" else False,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		weight_decay=args.weight_decay,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		dataloader_drop_last=True,
		eval_strategy="no",
		remove_unused_columns=True,
		seed=args.seed
	)

def main():
	args = parse_args()
	setup_environment()
	set_seed(args.seed)

	# Hugging Face 로그인
	if args.hf_token:
		login(args.hf_token)

	device = get_device()
	tokenizer, model = load_tokenizer_and_model(args, device)
	train_ds = prepare_dataset(args)

	# 데이터 전처리
	preprocess_function = create_preprocess_function(tokenizer, args)
	train_ds = train_ds.map(
		preprocess_function,
		batched=True,
		remove_columns=train_ds.column_names,
		desc="Preprocessing dataset"
	)

	# 학습 설정
	training_args = get_training_args(args, device)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	# Trainer 초기화 및 학습
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		data_collator=data_collator,
	)

	try:
		print("\nStarting training...")
		trainer.train()

		print("\nSaving model and tokenizer...")
		trainer.save_model(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)
		print(f"Model and tokenizer saved to {args.output_dir}")

	except Exception as e:
		print(f"\nError during training: {str(e)}")
		raise

if __name__ == "__main__":
	main()