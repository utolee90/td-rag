import os
os.environ["HF_DATASETS_CACHE"]="/mnt/nvme02/home/tdrag/.cache"
os.environ["TRANSFORMERS_CACHE"]="/mnt/nvme02/home/tdrag/.cache"

# create_indexes.py
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
import faiss
from uuid import uuid4
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import psutil
import time
import argparse
import numpy as np

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

# 저장
FAISS_PATH = "faiss_indexes"
faiss.write_index(index, f"{FAISS_PATH}/faiss_index.idx")



# 로깅 설정
logging.basicConfig(
    filename=f'index_creation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class IndexCreator:
    def __init__(self, base_dir='faiss_indexes', batch_size=100):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        '''torch.cuda.is_available()'''
        # 임베딩 모델 초기화                        'Vaiv/GeM2-Llamion-14B-Base',
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': False,'pad_token': '[PAD]'})#,kwargs={'pad_token':'[PAD]'}
        #)

        if 0:
            if self.embeddings.tokenizer.pad_token is None:
                self.embeddings.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.embeddings.model.resize_token_embeddings(len(self.embeddings.tokenizer))

    def process_file(self, file_path):
        try:
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            logging.info(f"Processing file: {file_path}")
            print(f"Processing file: {file_path}")

            # 데이터 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            print("LEN",len(news_data))
            # 데이터 형태 확인 
            if len(news_data) == 0:
                logging.warning("No data found in file")
                return
            elif not news_data[0].get('date') and news_data[0].get('item'):
                print("NEWSDATA")
                new_news_data = []
                for news_obj in news_data:
                    new_news_data.extend(news_obj['item']['documentList'])
                # news_data = news_data[0]['item']['documentList']
                news_data = new_news_data

            # 날짜별 그룹화
            date_groups = {}
            for news in tqdm(news_data, desc="Grouping by date"):
                date = news['date']
                if date not in date_groups:
                    date_groups[date] = []
                date_groups[date].append(news)

            # 날짜별 처리
            for date, articles in date_groups.items():
                # index_name = f"faiss_index_{date}"
                index_name = f"{date}"
                index_path = self.base_dir / index_name

                if index_path.exists():
                    logging.info(f"Index already exists for date {date}")
                    #continue

                logging.info(f"Creating index for date {date} with {len(articles)} articles")

                # 배치 처리
                for i in range(0, len(articles), self.batch_size):
                    batch = articles[i:i + self.batch_size]

                    if i == 0:  # 첫 배치에서 인덱스 초기화
                        documents = self.create_documents(batch)
                        dimension = len(self.embeddings.embed_query("hello world"))

                        # 기본 인덱스 생성
                        index = faiss.IndexFlatL2(dimension)
                        vector_store = FAISS(
                            embedding_function=self.embeddings,
                            index=index,
                            docstore=InMemoryDocstore(),
                            index_to_docstore_id={}
                        )
                    else:
                        documents = self.create_documents(batch)

                    # 문서 추가
                    uuids = [str(uuid4()) for _ in range(len(documents))]
                    vector_store.add_documents(documents=documents, ids=uuids)

                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    logging.info(f"Batch {i//self.batch_size + 1} processed. Memory usage: {current_memory:.2f}MB")

                # 인덱스 저장
                index_path.mkdir(exist_ok=True)
                vector_store.save_local(str(index_path))
                logging.info(f"Index saved for date {date}")

            elapsed_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_change = final_memory - initial_memory

            stats = f"""
            처리 완료:
            - 파일: {file_path}
            - 총 문서 수: {sum(len(articles) for articles in date_groups.values())}
            - 생성된 인덱스 수: {len(date_groups)}
            - 처리 시간: {elapsed_time:.2f}초
            - 메모리 사용량 변화: {memory_change:.2f}MB
            - 날짜별 문서 수: {', '.join(f'{k}: {len(v)}' for k, v in date_groups.items())}
            """
            logging.info(stats)
            print(stats)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            print(f"Error processing file {file_path}: {e}")


    def create_documents(self, news_data):
        documents = []
        for news in news_data:
            content = f"Title: {news['title']}\nContent: {news['content']}"
            metadata = {
                "date": news['date'],
                "docID": news['docID'],
                "url": news['url'],
                "source": news.get('writerRealName', ''),
                "projectId": news['projectId']
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents

def main():
    parser = argparse.ArgumentParser(description='Create FAISS indexes from news data')
    parser.add_argument('--input_dir', type=str, default='01_disaster_Fire_3years', help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='faiss_indexes', help='Output directory for indexes')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    args = parser.parse_args()

    creator = IndexCreator(base_dir=args.output_dir, batch_size=args.batch_size)

    # JSON 파일 처리
    input_path = Path(args.input_dir)
    json_files = list(input_path.glob('**/*.json'))

    print(f"Found {len(json_files)} JSON files")
    for file_path in json_files:
        creator.process_file(str(file_path))

if __name__ == "__main__":
    main()