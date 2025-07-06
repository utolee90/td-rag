from rank_bm25 import BM25Okapi
from typing import List, Dict, Generator
import json
import numpy as np

class BM25Retriever:
    def __init__(self, jsonl_path: str, batch_size: int = 1000):
        """
        Args:
            jsonl_path: JSONL 파일 경로
            batch_size: 한 번에 처리할 문서 수
        """
        self.jsonl_path = jsonl_path
        self.batch_size = batch_size
        self.total_docs = self._count_documents()
        self.bm25 = None
        self.documents = []
        
    def _count_documents(self) -> int:
        """파일의 총 문서 수를 계산"""
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def _document_generator(self) -> Generator[Dict, None, None]:
        """문서를 한 줄씩 읽어서 yield"""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    
    def _process_batch(self, batch: List[Dict]) -> List[str]:
        """배치의 문서들을 토큰화"""
        return [doc["text"].split() for doc in batch]
    
    def build_index(self):
        """BM25 인덱스 구축"""
        tokenized_docs = []
        current_batch = []
        
        # 문서를 배치 단위로 처리
        for doc in self._document_generator():
            current_batch.append(doc)
            
            if len(current_batch) >= self.batch_size:
                # 배치 처리
                tokenized_docs.extend(self._process_batch(current_batch))
                self.documents.extend(current_batch)
                current_batch = []
        
        # 남은 배치 처리
        if current_batch:
            tokenized_docs.extend(self._process_batch(current_batch))
            self.documents.extend(current_batch)
        
        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"Indexed {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """검색 수행"""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # 쿼리 토큰화
        tokenized_query = query.split()
        
        # BM25 점수 계산
        scores = self.bm25.get_scores(tokenized_query)
        
        # 상위 k개 문서 선택
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 결과 반환
        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "text": self.documents[idx]["text"],
                "score": float(scores[idx])
            })
        return results


def format_prompt(query, context):
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    

# 3. 사용 예시
if __name__ == "__main__":
    
    # BM25 리트리버 초기화
    retriever = BM25Retriever("enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl")
    # 인덱스 구축
    print("Building index...")
    retriever.build_index()
    
    # 검색 수행
    query = "What is the capital of France?"
    results = retriever.search(query, top_k=5)
    
    # 결과 출력
    for doc in results:
        print(f"Score: {doc['score']:.4f}")
        print(f"Title: {doc['title']}")
        print(f"Text: {doc['text'][:1000]}...\n")  # 텍스트는 처음 1000자만 출력
    
    print()
