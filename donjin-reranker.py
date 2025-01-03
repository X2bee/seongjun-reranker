import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_name_or_path, device):
	"""Load the model and tokenizer from the specified model name or path."""
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
	model.to(device)
	return tokenizer, model

def preprocess(query, candidates, tokenizer, max_len=512):
	"""Preprocess query and candidates for input into the model."""
	pairs = [[query, candidate] for candidate in candidates]
	encoded = tokenizer(
		text=[pair[0] for pair in pairs],
		text_pair=[pair[1] for pair in pairs],
		padding=True,
		truncation=True,
		max_length=max_len,
		return_tensors="pt"
	)
	return encoded

def rerank(query, candidates, model, tokenizer, device, max_len=512):
	"""Rerank candidates based on the query using the trained model."""
	inputs = preprocess(query, candidates, tokenizer, max_len)
	inputs = {key: val.to(device) for key, val in inputs.items()}

	with torch.no_grad():
		outputs = model(**inputs)
		scores = outputs.logits.squeeze()

	sorted_indices = torch.argsort(scores, descending=True)
	ranked_candidates = [candidates[i] for i in sorted_indices]
	return ranked_candidates, scores

def main():
	# Model name and device setup
	model_name_or_path = "Dongjin-kr/ko-reranker"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Load the model and tokenizer
	tokenizer, model = load_model_and_tokenizer(model_name_or_path, device)

	# Test data
	queries = [
		"서울 날씨는?",
		"맛집 추천",
		"주말에 갈만한 여행지 추천",
		"Python 설치 방법",
		"최근 인기 있는 영화는?"
	]
	candidates = [
		["서울의 오늘 날씨는 맑음", "어제 서울은 비가 옴", "서울은 오늘 흐리고 비가 예상됨"],
		["맛있는 피자집", "파스타 전문점 추천", "스시 맛집 리스트"],
		["강원도 속초 여행", "제주도 올레길 추천", "부산 해운대 방문기"],
		["Python 공식 홈페이지에서 다운로드", "Anaconda를 이용한 설치 방법", "Python 설치 오류 해결"],
		["2025년 아카데미 수상작", "흥행 중인 액션 영화", "인기 있는 로맨스 영화"]
	]

	# Test reranker
	for query, candidates_set in zip(queries, candidates):
		ranked, scores = rerank(query, candidates_set, model, tokenizer, device)
		print(f"\nQuery: {query}")
		for rank, (candidate, score) in enumerate(zip(ranked, scores), 1):
			print(f"{rank}: {candidate} (Score: {score.item():.4f})")

if __name__ == "__main__":
	main()