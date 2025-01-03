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
		"스니커즈 추천",
		"겨울 코트",
		"노트북 가방",
		"무선 이어폰",
		"홈 카페 머신"
	]

	candidates = [
		[
			"나이키 에어포스 1", "아디다스 스탠 스미스", "반스 올드스쿨", "컨버스 척테일러", "뉴발란스 574",
			"리복 클래식", "퓨마 스웨이드", "아식스 젤라이트", "언더아머 커리", "조던 레트로 1",
			"발렌시아가 트리플 S", "구찌 라이톤", "프라다 클라우드버스트", "디올 오블리크", "생로랑 코트",
			"필라 디스럽터", "라코스테 스니커즈", "톰스 알파인", "스케쳐스 딜라이트", "버켄스탁 스니커즈",
			"에코 바이옴", "미즈노 웨이브", "홀리데이 아웃도어", "솔로베어 스니커즈", "머렐 스니커즈",
			"콜한 제로그랜드", "락포트 스니커즈", "크록스 라이트라이드", "언더아머 호버", "알엠윌리엄스 부츠",
			"테바 허리케인", "나이키 덩크 로우", "나이키 조던 11", "스케쳐스 고워크", "피엘라벤 Kanken",
			"아식스 타이거", "살로몬 스니커즈", "데카트론 킵런", "캠퍼 스니커즈", "허쉬 스니커즈",
			"안야힌드마치 스니커즈", "MLB 빅볼청키", "버버리 스니커즈", "C.P.컴퍼니", "톰포드 스니커즈",
			"알렉산더 맥퀸", "오프화이트 스니커즈", "이세이 미야케 스니커즈", "Y-3 스니커즈", "BAPE 스니커즈",
			"삼성 갤럭시 스마트폰", "LG 그램 노트북", "드롱기 커피머신", "샤오미 공기청정기", "닌텐도 스위치"
		],
		[
			"캐시미어 롱 코트", "숏 패딩 코트", "울 블렌드 코트", "트렌치 코트", "더블 브레스티드 코트",
			"싱글 브레스티드 코트", "체스터필드 코트", "퓨어 캐시미어 코트", "오버사이즈 코트", "테일러드 코트",
			"벨티드 코트", "코쿤 코트", "카멜 코트", "블랙 롱 코트", "화이트 코트",
			"라이트 그레이 코트", "네이비 블루 코트", "아이보리 코트", "크림색 코트", "브라운 코트",
			"체크 패턴 코트", "하운드투스 코트", "애니멀 프린트 코트", "컬러 블록 코트", "지퍼 디테일 코트",
			"퍼 트림 코트", "리버시블 코트", "퀼팅 코트", "후드 롱 코트", "사파리 코트",
			"드롭 숄더 코트", "패치 포켓 코트", "더플 코트", "시어링 코트", "수트 스타일 코트",
			"폴리에스터 코트", "모직 코트", "패딩 믹스 코트", "캐주얼 코트", "미디 코트",
			"클래식 코트", "빈티지 코트", "모노크롬 코트", "데님 코트", "벨벳 코트",
			"라펠 코트", "프릴 디테일 코트", "플리스 코트", "자켓 스타일 코트", "하이넥 코트",
			"아이폰 15", "소니 플레이스테이션 5", "샤오미 로봇청소기", "브레빌 토스터", "삼성 QLED TV"
		],
		[
			"15인치 노트북 백팩", "슬림 노트북 서류 가방", "방수 노트북 가방", "멀티 포켓 노트북 백팩", "가죽 노트북 가방",
			"메신저 스타일 가방", "크로스바디 노트북 가방", "미니멀 노트북 케이스", "패딩 노트북 슬리브", "하드 케이스 노트북 가방",
			"충격 흡수 노트북 가방", "지퍼 포켓 노트북 가방", "비즈니스 노트북 가방", "캐주얼 노트북 백팩", "게이밍 노트북 백팩",
			"컴팩트 노트북 가방", "이동형 노트북 케이스", "에코 프렌들리 가방", "슬링백 노트북 가방", "블랙 노트북 가방",
			"레더 스타일 노트북 가방", "2-in-1 노트북 가방", "캐리어 스타일 가방", "힙색 노트북 가방", "데이팩 노트북 가방",
			"롤탑 노트북 백팩", "토트백 스타일 가방", "클래식 노트북 가방", "포멀 스타일 가방", "스포티 노트북 가방",
			"LED 백라이트 가방", "언더아머 노트북 가방", "아디다스 노트북 백팩", "나이키 노트북 백팩", "텀블러 포함 가방",
			"멀티미디어 포켓 가방", "서류형 노트북 가방", "페브릭 노트북 가방", "맥북 전용 가방", "게이밍 노트북 슬리브",
			"스마트 락 가방", "항균 처리 노트북 가방", "방진 처리 노트북 가방", "라이트웨이트 가방", "패턴 디자이너 가방",
			"라벨 디테일 가방", "하드쉘 노트북 케이스", "퍼플 노트북 가방", "그린 노트북 가방", "화이트 노트북 가방",
			"닌텐도 스위치", "삼성 갤럭시 탭", "애플 아이패드", "BOSE 블루투스 스피커", "다이슨 무선청소기"
		],
		[
			"애플 에어팟 프로", "삼성 갤럭시 버즈 2", "소니 WF-1000XM4", "BOSE QC 이어버드", "젠하이저 모멘텀 트루 와이어리스",
			"Jabra Elite 85t", "Anker Soundcore Liberty Air 2", "Shure Aonic 215", "Bang & Olufsen Beoplay E8", "LG 톤 프리",
			"Google Pixel Buds", "Amazon Echo Buds", "Beats Fit Pro", "Marshall Minor III", "Jaybird Vista 2",
			"Plantronics BackBeat Pro", "Skullcandy Indy Evo", "Razer Hammerhead", "Edifier TWS NB2", "Huawei FreeBuds",
			"Xiaomi Redmi Earbuds", "Realme Buds Air", "OnePlus Buds Pro", "Soundpeats Sonic", "Vivo TWS Earbuds",
			"OPPO Enco X", "Aukey EP-N5", "Tronsmart Apollo Bold", "Cleer Ally Plus II", "Master & Dynamic MW08",
			"Creative Outlier Air V3", "Sabbat X12 Ultra", "TOZO NC2", "Enacfire G20", "Vankyo X400",
			"Soul Sync Pro", "Lypertek PurePlay Z3", "KZ ZS10 Pro", "Moondrop Sparks", "Grado GT220",
			"Earfun Free Pro", "Hifiman TWS600", "Status Audio BT One", "Astell & Kern AK UW100", "Fiil T1X",
			"Lenovo True Wireless", "Meizu POP Pro", "Haylou GT5", "BlitzWolf BW-FYE8", "Urbanista London",
			"소니 알파 카메라", "DJI 미니 드론", "MSI 게이밍 노트북", "애플 매직 키보드", "나이키 런닝화"
		],
		[
			"브레빌 커피 머신", "드롱기 전자동 머신", "필립스 라떼고", "네스프레소 버츄오", "커피빈 캡슐 머신",
			"쿠진아트 에스프레소 머신", "샤오미 커피 머신", "가찌아 클래식 프로", "라 심발리 에스프레소 머신", "라 마르조코 홈",
			"드롱기 아이코나 빈티지", "제너럴 커피 머신", "WMF 퍼펙트 밀크", "몰타저 커피 머신", "유라 S8",
			"세코 빈투컵 머신", "테팔 필터 커피 머신", "마샬드 커피 머신", "HARIO V60", "모카마스터",
			"키친에이드 브루어", "칼리타 커피 드립", "비스카 커피 머신", "루멕스 커피 머신", "Zojirushi 커피 메이커",
			"타워 브랜드 머신", "Bonavita 브루어", "레코머 전자동 머신", "테크니폼 드리퍼", "커피트로닉 머신",
			"Eureka 전자동 머신", "Lelit Bianca 머신", "Ninja 커피 바", "Bonavita 커피 바", "Behmor 브루어",
			"Brim 커피 머신", "Rancilio Silvia 머신", "La Pavoni 머신", "Casadio Undici 머신", "Miele 커피 머신",
			"Rocket Appartamento", "Profitec Pro 300", "Synesso ES-1", "Decent Espresso DE1", "Espazzola 세척기",
			"Wacaco Minipresso", "Handpresso 펌프", "Staresso Pro", "Flair Espresso", "Cafflano Kompresso",
			"LG 스타일러", "삼성 비스포크 냉장고", "필립스 전기 면도기", "다이슨 에어랩", "고프로 히어로 카메라"
		]
	]


	# Test reranker
	for query, candidates_set in zip(queries, candidates):
		ranked, scores = rerank(query, candidates_set, model, tokenizer, device)
		print(f"\nQuery: {query}")
		for rank, (candidate, score) in enumerate(zip(ranked, scores), 1):
			print(f"{rank}: {candidate} (Score: {score.item():.4f})")

if __name__ == "__main__":
	main()