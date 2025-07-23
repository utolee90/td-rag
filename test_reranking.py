"""
Reranking 테스트 및 사용 예제
"""

from reranker import SearchReranker, rerank_with_cohere, rerank_with_custom_scoring
from keys import COHERE_API_KEY, OPENAI_API_KEY

def test_reranking():
    """Reranking 기능 테스트"""
    
    # 샘플 검색 결과
    sample_query = "What happened in the fire incident?"
    sample_results = [
        {
            "doc_id": "doc1",
            "title": "Building Fire in Downtown",
            "text": "A fire broke out in a downtown building yesterday, causing significant damage...",
            "url": "https://example.com/fire1",
            "publish_date": "20250721"
        },
        {
            "doc_id": "doc2", 
            "title": "Weather Report",
            "text": "Today's weather forecast shows sunny skies with temperatures reaching...",
            "url": "https://example.com/weather",
            "publish_date": "20250722"
        },
        {
            "doc_id": "doc3",
            "title": "Fire Department Response",
            "text": "The local fire department responded quickly to the emergency call...",
            "url": "https://example.com/fire2",
            "publish_date": "20250721"
        }
    ]
    
    print("=== 원본 검색 결과 ===")
    for i, result in enumerate(sample_results):
        print(f"{i+1}. {result['title']}")
    
    # 1. Custom reranking 테스트
    print("\n=== Custom Reranking 결과 ===")
    custom_reranked = rerank_with_custom_scoring(sample_query, sample_results, top_k=3)
    for i, result in enumerate(custom_reranked):
        score = result.get('rerank_score', 0)
        print(f"{i+1}. {result['title']} (score: {score:.2f})")
    
    # 2. Cohere reranking 테스트 (API 키가 있는 경우)
    if COHERE_API_KEY:
        print("\n=== Cohere Reranking 결과 ===")
        try:
            cohere_reranked = rerank_with_cohere(sample_query, sample_results, COHERE_API_KEY, top_k=3)
            for i, result in enumerate(cohere_reranked):
                score = result.get('rerank_score', 0)
                print(f"{i+1}. {result['title']} (score: {score:.3f})")
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
    else:
        print("\n=== Cohere API 키가 설정되지 않아 건너뜀 ===")
    
    # 3. OpenAI reranking 테스트 (API 키가 있는 경우)
    if OPENAI_API_KEY:
        print("\n=== OpenAI Reranking 결과 ===")
        try:
            from reranker import rerank_with_openai
            openai_reranked = rerank_with_openai(sample_query, sample_results, OPENAI_API_KEY, top_k=3)
            for i, result in enumerate(openai_reranked):
                score = result.get('rerank_score', 0)
                print(f"{i+1}. {result['title']} (score: {score:.3f})")
        except Exception as e:
            print(f"OpenAI reranking failed: {e}")


def compare_reranking_methods():
    """다양한 reranking 방법 비교"""
    
    query = "Samsung Galaxy phone release"
    results = [
        {"title": "Apple iPhone 15 Review", "text": "The new Apple iPhone 15 features advanced camera...", "doc_id": "1"},
        {"title": "Samsung Galaxy S24 Launch", "text": "Samsung announced the Galaxy S24 with AI features...", "doc_id": "2"}, 
        {"title": "Google Pixel 8 Pro", "text": "Google's latest Pixel 8 Pro smartphone comes with...", "doc_id": "3"},
        {"title": "Samsung Galaxy Features", "text": "The Samsung Galaxy series continues to innovate...", "doc_id": "4"},
    ]
    
    print(f"쿼리: {query}\n")
    
    print("=== 원본 순서 ===")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
    
    print("\n=== Custom Reranking ===")
    custom_ranked = rerank_with_custom_scoring(query, results, top_k=4)
    for i, result in enumerate(custom_ranked):
        score = result.get('rerank_score', 0)
        print(f"{i+1}. {result['title']} (score: {score:.2f})")


if __name__ == "__main__":
    print("🔍 Reranking 기능 테스트 시작\n")
    
    test_reranking()
    print("\n" + "="*50 + "\n")
    compare_reranking_methods()
    
    print("\n✅ 테스트 완료")
    
    # 사용법 안내
    print("""
📖 사용법:

1. Custom reranking (API 키 불필요):
   from reranker import rerank_with_custom_scoring
   results = rerank_with_custom_scoring(query, search_results, top_k=10)

2. Cohere reranking (추천):
   from reranker import rerank_with_cohere
   results = rerank_with_cohere(query, search_results, cohere_api_key, top_k=10)

3. OpenAI reranking:
   from reranker import rerank_with_openai
   results = rerank_with_openai(query, search_results, openai_api_key, top_k=10)

4. Cross-encoder reranking:
   from reranker import rerank_with_cross_encoder
   results = rerank_with_cross_encoder(query, search_results, top_k=10)

⚙️ 설정:
- Cohere API 키를 환경변수 COHERE_API_KEY에 설정하세요
- OpenAI API 키는 이미 OPENAI_API_KEY에 설정되어 있습니다
- Cross-encoder는 추가 모델 다운로드가 필요합니다
""")
