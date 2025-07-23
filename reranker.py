"""
Reranking utilities for search results
"""
import cohere
import openai
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import torch
from functools import lru_cache

class SearchReranker:
    """검색 결과를 reranking하는 클래스"""
    
    def __init__(self, method="cohere", api_key=None, model_name=None):
        """
        Args:
            method: "cohere", "openai", "cross_encoder", "custom"
            api_key: API 키 (cohere 또는 openai용)
            model_name: 모델 이름 (cross_encoder용)
        """
        self.method = method
        self.api_key = api_key
        
        if method == "cohere":
            if not api_key:
                raise ValueError("Cohere API key is required")
            self.client = cohere.Client(api_key)
            
        elif method == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.client = openai.OpenAI(api_key=api_key)
            
        elif method == "cross_encoder":
            model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.cross_encoder = CrossEncoder(model_name)
            
        self.method = method
    
    def rerank_gcs_results(self, query: str, search_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        GCS 검색 결과를 reranking
        
        Args:
            query: 검색 쿼리
            search_results: GCS 검색 결과 리스트
            top_k: 반환할 상위 결과 개수
            
        Returns:
            reranking된 검색 결과 리스트
        """
        if not search_results:
            return []
            
        if self.method == "cohere":
            return self._rerank_with_cohere(query, search_results, top_k)
        elif self.method == "openai":
            return self._rerank_with_openai(query, search_results, top_k)
        elif self.method == "cross_encoder":
            return self._rerank_with_cross_encoder(query, search_results, top_k)
        elif self.method == "custom":
            return self._rerank_with_custom_scoring(query, search_results, top_k)
        else:
            # 기본적으로 원래 순서 유지
            return search_results[:top_k]
    
    def _rerank_with_cohere(self, query: str, search_results: List[Dict], top_k: int) -> List[Dict]:
        """Cohere Rerank API를 사용한 reranking"""
        try:
            # 문서 텍스트 준비 (title + snippet + text)
            documents = []
            for result in search_results:
                doc_text = ""
                if result.get("title"):
                    doc_text += f"Title: {result['title']}\n"
                if result.get("text"):
                    doc_text += f"Content: {result['text'][:1000]}..."  # 첫 1000자만
                elif result.get("snippet"):
                    doc_text += f"Snippet: {result['snippet']}"
                    
                documents.append(doc_text.strip())
            
            if not documents:
                return search_results[:top_k]
            
            max_chunks = min(top_k, len(documents))
            
            # Cohere Rerank API 호출
            response = self.client.rerank(
                model="rerank-multilingual-v3.0",  # 또는 "rerank-english-v3.0"
                query=query,
                documents=documents,
                return_documents=False
            )
            
            # 결과 재정렬
            reranked_results = []
            for result in response.results:
                original_idx = result.index
                score = result.relevance_score
                
                # 원본 결과에 점수 추가
                reranked_result = search_results[original_idx].copy()
                reranked_result['rerank_score'] = score
                reranked_results.append(reranked_result)
            
            print(f"✅ Cohere reranking: {len(reranked_results)} results")
            return reranked_results[:max_chunks]
            
        except Exception as e:
            print(f"⚠️ Cohere reranking failed: {e}")
            return search_results[:top_k]
    
    def _rerank_with_openai(self, query: str, search_results: List[Dict], top_k: int) -> List[Dict]:
        """OpenAI를 사용한 LLM 기반 reranking"""
        try:
            # 결과를 번호와 함께 텍스트로 준비
            documents_text = ""
            for i, result in enumerate(search_results):
                title = result.get("title", "No title")
                text = result.get("text", result.get("snippet", "No content"))[:500]
                documents_text += f"{i+1}. Title: {title}\nContent: {text}\n\n"
            
            prompt = f"""Given the query "{query}", please rank the following search results by relevance. 
Return only the numbers (1-{len(search_results)}) in order of most relevant to least relevant, separated by commas.

Search Results:
{documents_text}

Ranking (most relevant first):"""
            
            max_chunks = min(top_k, len(search_results))

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            # 순서 파싱
            ranking_text = response.choices[0].message.content.strip()
            try:
                rankings = [int(x.strip()) - 1 for x in ranking_text.split(',')]
                rankings = [r for r in rankings if 0 <= r < len(search_results)]
            except:
                rankings = list(range(len(search_results)))
            
            # 결과 재정렬
            reranked_results = []
            for rank, original_idx in enumerate(rankings[:top_k]):
                result = search_results[original_idx].copy()
                result['rerank_score'] = 1.0 - (rank / len(rankings))  # 순위를 점수로 변환
                reranked_results.append(result)
            
            print(f"✅ OpenAI reranking: {len(reranked_results)} results")
            return reranked_results[:max_chunks]
            
        except Exception as e:
            print(f"⚠️ OpenAI reranking failed: {e}")
            return search_results[:top_k]
    
    def _rerank_with_cross_encoder(self, query: str, search_results: List[Dict], top_k: int) -> List[Dict]:
        """Cross-encoder 모델을 사용한 reranking"""
        try:
            # 쿼리-문서 쌍 준비
            query_doc_pairs = []
            for result in search_results:
                doc_text = ""
                if result.get("title"):
                    doc_text += result["title"] + " "
                if result.get("text"):
                    doc_text += result["text"][:1000]
                elif result.get("snippet"):
                    doc_text += result["snippet"]
                    
                query_doc_pairs.append([query, doc_text.strip()])
            
            if not query_doc_pairs:
                return search_results[:top_k]
            
            max_chunks = min(top_k, len(query_doc_pairs))
            
            # Cross-encoder로 점수 계산
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # 점수와 함께 결과 정렬
            scored_results = []
            for i, score in enumerate(scores):
                result = search_results[i].copy()
                result['rerank_score'] = float(score)
                scored_results.append(result)
            
            # 점수순으로 정렬
            scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            print(f"✅ Cross-encoder reranking: {len(scored_results[:max_chunks])} results")
            return scored_results[:max_chunks]
            
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            return search_results[:top_k]
    
    def _rerank_with_custom_scoring(self, query: str, search_results: List[Dict], top_k: int) -> List[Dict]:
        """커스텀 스코어링을 사용한 간단한 reranking"""
        try:
            query_terms = set(query.lower().split())

            max_chunks = min(top_k, len(search_results))
            
            scored_results = []
            for result in search_results:
                score = 0.0
                
                # Title에서 쿼리 용어 매칭 (가중치 2.0)
                title = result.get("title", "").lower()
                title_matches = sum(1 for term in query_terms if term in title)
                score += title_matches * 2.0
                
                # Content/snippet에서 쿼리 용어 매칭 (가중치 1.0)
                content = result.get("text", result.get("snippet", "")).lower()
                content_matches = sum(1 for term in query_terms if term in content)
                score += content_matches * 1.0
                
                # 날짜 기반 점수 (최신일수록 높은 점수)
                publish_date = result.get("publish_date", "20000101")
                try:
                    date_score = int(publish_date) / 20250101  # 정규화
                    score += date_score * 0.5
                except:
                    pass
                
                result_copy = result.copy()
                result_copy['rerank_score'] = score
                scored_results.append(result_copy)
            
            # 점수순으로 정렬
            scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            print(f"✅ Custom reranking: {len(scored_results[:max_chunks])} results")
            return scored_results[:max_chunks]
            
        except Exception as e:
            print(f"⚠️ Custom reranking failed: {e}")
            return search_results[:top_k]


# 편의 함수들
def rerank_with_cohere(query: str, search_results: List[Dict], api_key: str, top_k: int = 10) -> List[Dict]:
    """Cohere를 사용한 간단한 reranking 함수"""
    reranker = SearchReranker("cohere", api_key=api_key)
    return reranker.rerank_gcs_results(query, search_results, top_k)

def rerank_with_openai(query: str, search_results: List[Dict], api_key: str, top_k: int = 10) -> List[Dict]:
    """OpenAI를 사용한 간단한 reranking 함수"""
    reranker = SearchReranker("openai", api_key=api_key)
    return reranker.rerank_gcs_results(query, search_results, top_k)

def rerank_with_cross_encoder(query: str, search_results: List[Dict], model_name: str = None, top_k: int = 10) -> List[Dict]:
    """Cross-encoder를 사용한 간단한 reranking 함수"""
    reranker = SearchReranker("cross_encoder", model_name=model_name)
    return reranker.rerank_gcs_results(query, search_results, top_k)

def rerank_with_custom_scoring(query: str, search_results: List[Dict], top_k: int = 10) -> List[Dict]:
    """커스텀 스코어링을 사용한 간단한 reranking 함수"""
    reranker = SearchReranker("custom")
    return reranker.rerank_gcs_results(query, search_results, top_k)
