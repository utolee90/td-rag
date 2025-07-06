import argparse
import requests, jsonlines
from newspaper import Article
import datetime
from retrieval_utils.tools import read_jsonl
import json
import datetime
import concurrent.futures
import time
from functools import lru_cache
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# urllib3 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_gcs(key,
           engine,
           in_file,
           out_file,
           use_evidence=False,
           max_workers=4  # 병렬 처리 워커 수 추가
          ):
    questions = read_jsonl(in_file)
    outputs = []
    
    print(f"Processing {len(questions)} questions with {max_workers} workers...")
    start_time = time.time()
    
    for question in questions:
        if use_evidence:
            search_result = search(question["evidence"], key, engine)
        else:
            search_result = search(question["question_sentence"], key, engine)
        search_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")
        
        # 병렬로 기사 파싱 처리
        search_result = parse_articles_parallel(search_result, max_workers)
        
        output = {"question_id": question["question_id"], "search_time": search_time, "search_result": search_result}
        outputs.append(output)
    
    processing_time = time.time() - start_time
    print(f"✅ GCS processing completed in {processing_time:.2f} seconds")
    
    return outputs

def search(query, key, engine, top_k=10, timeout=10):
    """Google Custom Search API 호출 (타임아웃 및 에러 처리 개선)"""
    url = f"https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}&start=1"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # HTTP 에러 체크
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ 검색 API 호출 실패: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 실패: {str(e)}")
        return []
    
    search_items = data.get("items", [])
    results = []
    
    for i, search_item in enumerate(search_items):
        try:
            long_description = search_item["pagemap"]["metatags"][0]["og:description"]
        except KeyError:
            long_description = "N/A"
            
        doc_id = f'search_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{query[:25]}_{i}'
        title = search_item.get("title", "Unknown Title")
        snippet = search_item.get("snippet", "")
        html_snippet = search_item.get("htmlSnippet", "")
        link = search_item.get("link")
        
        if link:  # URL이 있는 경우만 추가
            result = {"url": link, "title": title, "doc_id": doc_id}
            results.append(result)
            
        if len(results) >= top_k:
            break

    print(f"🔍 검색 완료: '{query[:30]}...' - {len(results)}개 결과")
    return results

@lru_cache(maxsize=128)  # URL별 캐싱으로 중복 파싱 방지
def parse_article(url, timeout=15):
    """기사 파싱 (캐싱 및 타임아웃 적용, 강화된 에러 처리)"""
    try:
        # 강력한 세션 사용
        session = get_session()
        
        # Article 객체 생성 (간단한 방식으로 수정)
        article = Article(url)
        
        # requests_session을 설정할 수 있는지 확인
        if hasattr(article, 'set_session'):
            article.set_session(session)
        elif hasattr(article, 'session'):
            article.session = session
            
        # 타임아웃 설정을 위한 다양한 방법 시도
        try:
            article.download()
        except Exception as download_error:
            print(f"⚠️ 다운로드 실패, 대체 방법 시도: {download_error}")
            # 직접 requests로 콘텐츠 가져오기
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            article.set_html(response.text)
        
        article.parse()
        
        text = article.text or ""
        authors = article.authors or []
        
        # Wikipedia 특별 처리
        if 'wikipedia' in url.lower():
            authors = ["Wikipedia"]
        
        publish_date = article.publish_date
        if publish_date is not None:
            publish_date = publish_date.strftime("%Y/%m/%d")
        else:
            publish_date = "2018/12/31"
            
        # 텍스트가 너무 짧으면 다른 속성들 시도
        if len(text.strip()) < 50:
            fallback_text = ""
            
            # 안전하게 다양한 속성들 시도
            for attr in ['meta_description', 'summary', 'meta_data']:
                try:
                    value = getattr(article, attr, None)
                    if value:
                        if isinstance(value, dict):
                            fallback_text = value.get('description', '') or value.get('summary', '')
                        else:
                            fallback_text = str(value)
                        if fallback_text.strip():
                            break
                except:
                    continue
            
            text = fallback_text or text
            
        return text, authors, publish_date
        
    except Exception as e:
        print(f"⚠️ 기사 파싱 실패 ({url}): {str(e)}")
        return "", [], "2018/12/31"

def parse_articles_parallel(search_result, max_workers=4):
    """병렬로 기사들을 파싱하여 성능 향상"""
    if not search_result:
        return search_result
    
    def parse_single_article(article):
        """단일 기사 파싱 (병렬 처리용)"""
        try:
            article["text"], article["authors"], article["publish_date"] = parse_article(article["url"])
            return article
        except Exception as e:
            print(f'\nURL 파싱 실패: {article["url"]} - {str(e)}\n')
            # 실패한 기사는 빈 데이터로 처리
            article["text"], article["authors"], article["publish_date"] = "", [], "2018/12/31"
            return article
    
    # 병렬 처리로 기사들 파싱
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        parsed_articles = list(executor.map(parse_single_article, search_result))
    
    # 유효한 기사들만 필터링 (텍스트가 있는 것들)
    valid_articles = [article for article in parsed_articles if article["text"].strip()]
    
    print(f"✅ 파싱 완료: {len(valid_articles)}/{len(search_result)} 기사 성공")
    return valid_articles

def clear_parse_cache():
    """파싱 캐시 클리어 및 세션 정리 (메모리 관리용)"""
    global _session
    
    parse_article.cache_clear()
    
    # 세션도 정리
    if _session:
        _session.close()
        _session = None
        
    print("✅ 기사 파싱 캐시와 세션이 클리어되었습니다.")

def get_cache_info():
    """캐시 사용 통계 반환"""
    cache_info = parse_article.cache_info()
    print(f"📊 캐시 통계 - 히트: {cache_info.hits}, 미스: {cache_info.misses}, 크기: {cache_info.currsize}/{cache_info.maxsize}")
    return cache_info

def create_robust_session(timeout=15):
    """강력한 HTTP 세션 생성 (재시도 및 타임아웃 설정)"""
    session = requests.Session()
    
    # 재시도 전략 설정
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    
    # HTTP 어댑터에 재시도 전략 적용
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 기본 헤더 설정
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    return session

# 전역 세션 인스턴스
_session = None

def get_session():
    """전역 세션 인스턴스 가져오기"""
    global _session
    if _session is None:
        _session = create_robust_session()
    return _session
