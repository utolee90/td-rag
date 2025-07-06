import os
os.environ["HF_DATASETS_CACHE"]="/mnt/nvme02/User/utopiamath/.cache"
os.environ["TRANSFORMERS_CACHE"]="/mnt/nvme02/User/utopiamath/.cache"

# create_indexes.py
import torch
import faiss
from uuid import uuid4

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from datasets import load_from_disk

# Load DPR encoder
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# 예시 문서 리스트
documents = ["Hello world.", "Paris is the capital of France.", "The Great Wall is in China."]

# 토크나이즈 및 임베딩 생성
inputs = ctx_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embeddings = ctx_encoder(**inputs).pooler_output  # (batch_size, hidden_dim)



# numpy 배열로 변환
embeddings_np = embeddings.cpu().numpy()

# FAISS index 생성 (cosine 유사도와 유사한 L2 거리)
dim = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dim)

# 인덱스에 추가
index.add(embeddings_np)

# 문서와의 매핑을 위해 ID 저장
doc_id_map = {i: doc for i, doc in enumerate(documents)}

# 쿼리 인코더 로드
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# 쿼리 예시
query = "Where is the Great Wall?"

# 쿼리 인코딩
q_inputs = question_tokenizer(query, return_tensors="pt")
with torch.no_grad():
    q_embedding = question_encoder(**q_inputs).pooler_output  # (1, dim)

# 검색
q_np = q_embedding.cpu().numpy()
D, I = index.search(q_np, k=3)  # top-3 검색

# 결과 출력
for rank, idx in enumerate(I[0]):
    print(f"Rank {rank+1}: {doc_id_map[idx]} (Score: {D[0][rank]})")

