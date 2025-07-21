import json, os, pickle, re
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from uuid import uuid4
from tqdm import tqdm
import time
import datetime
import openai

from retrieval.dpr import run_dpr_question, load_model
from retrieval.gcs import search as gcs_search, parse_article

class DocumentV2(BaseModel):
    """Pydantic v2 호환 Document 클래스"""
    metadata: Dict[str, Any]
    page_content: str

    @classmethod
    def from_v1(cls, data: dict):
        """Pydantic v1 객체를 v2로 변환"""
        if "__dict__" in data:
            data = data["__dict__"]  # ✅ Pydantic v1의 __dict__ 추출
        
        # ✅ 필수 필드가 존재하지 않으면 기본값 제공
        metadata = data.get("metadata", {})
        page_content = data.get("page_content", "")

        return cls(metadata=metadata, page_content=page_content)

class FakeLangChainObject:
    """ 기존 LangChain 객체 대신 사용할 더미 클래스 """
    def __init__(self, data):
        self.__dict__ = data  # 원래 객체처럼 __dict__ 속성을 유지

    def __repr__(self):
        return f"FakeLangChainObject({self.__dict__})"

    def to_pydantic_v2(self):
        """Pydantic v1 객체를 Pydantic v2로 변환"""
        if "__dict__" in self.__dict__:
            raw_data = self.__dict__["__dict__"]
        else:
            raw_data = self.__dict__

        # ✅ 불필요한 Pydantic v1 내부 필드 제거
        raw_data.pop("__pydantic_extra__", None)
        raw_data.pop("__pydantic_fields_set__", None)
        raw_data.pop("__pydantic_private__", None)

        return DocumentV2.from_v1(raw_data)  # ✅ Pydantic v2 객체 변환

class SafeUnpickler(pickle.Unpickler):
    """ 특정 모듈 오류를 무시하고 안전하게 pickle 데이터를 불러오는 클래스 """
    def find_class(self, module, name):
        # if module in ["langchain_community.docstore.in_memory", "langchain_core.documents.base"]:
        #    print(f"⚠️ 경고: `{module}.{name}`을(를) 찾을 수 없음. 더미 클래스로 대체")
        #    return FakeLangChainObject  # ✅ dict 대신 더미 클래스 사용
        return super().find_class(module, name)


class MergedDataV2(BaseModel):
    """변환된 Pydantic v2 데이터 + ID 매핑을 포함한 모델"""
    document: DocumentV2
    id_mapping: Dict[int, str]


def clean_dict(data):
    keys_to_remove = ['writerName', 'writerCodeString', 'vksDlOnly', 'countMap']
    cleaned_data = data.copy()
    for key in keys_to_remove:
        cleaned_data.pop(key, None)
    return cleaned_data

def clean_json_file_and_save_new(path):
    with open(path, 'r', encoding='utf-8') as f:
        content_list_one_json = json.load(f)
    final_list = []
    for idx,content in enumerate(content_list_one_json):
        print("idx", idx)
        list_sing = content["item"]["documentList"]
        final_list.extend(list_sing)

    final_list_v2=[]
    for idx,content in enumerate(final_list):
        print("idx", idx)
        if "#@VK#S1#사회" not in content["vks"]:
            continue
        cleaned_dict = clean_dict(content)
        final_list_v2.append(cleaned_dict)

    new_path = path.replace('/news_data_split_small','/news_data_split_small_V2')
    new_path_dir = '/'.join(new_path.split('/')[:-1])
    print("new_path_dir", new_path_dir)
    os.makedirs(new_path_dir, exist_ok=True)

    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(final_list_v2, f, ensure_ascii=False, indent=4)

def load_news_data(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.json'):       
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. .jsonl 또는 .json 파일을 사용하세요.")

def group_news_data_by_date(news_data):
    """뉴스 데이터를 날짜별로 그룹화"""
    grouped_data = {}
    for news in news_data:
        date = news.get('date', news.get('question_date', news.get('publish_date', '20000000'))).replace('/', '').replace('-', '')[:8] # 8자리 숫자 처리
        if date not in grouped_data:
            grouped_data[date] = []
        grouped_data[date].append(news)
    return grouped_data

def join_search_data(news_data):
    new_news_data = []
    for result in news_data:
        search_result = result.get('search_result', [])
        new_news_data.extend(search_result)
    
    return new_news_data


# 문서 생성 함수
def create_documents(news_data, max_length=4000):
    documents = []
    # search_data = news_data['search_result'] # structure news_data key(search_result)
    # create documents 데이터 형식에 따라 처리
    if isinstance(news_data, list):
        sample_news = news_data[0] if news_data else {}
        if 'search_result' in sample_news: #search_result 키가 있는 경우
            print("[DEBUG] 'search_result' key found in news_data, joining search data.")
            search_data = join_search_data(news_data)
        else: # news_data가 텍스트 정보 직접 포함
            search_data = news_data
    elif isinstance(news_data, dict):
        if 'search_result' in news_data:
            search_data = news_data['search_result']
        else:
            raise ValueError("Not expected form for 'news_data'. They key 'search_result' is missing.")

    print("[DEBUG] search_data type:", type(search_data))
    new_time = time.time()
    for news in tqdm(search_data):
        print("[DEBUG] Processing news item:", news.keys())
        # news가 dict인지 확인
        if isinstance(news, FakeLangChainObject):
            news = news.to_pydantic_v2().dict()
        if news.get('text', None) is None and news.get('content', None) is None:
            print("[DEBUG] news text or content is None, skipping this news item.")
            continue
        content = news.get('text', news.get('content', ''))
        title = content.split (' / ')[0] if ' / ' in content else news.get('title', 'No Title')
        text_content = content.split(' / ')[1] if ' / ' in content else content
        # 텍스트 내용이 너무 길면 잘라내기
        if len(text_content) > max_length:
            text_content = text_content[:max_length] + "..."
        content = f"Title: {title}\nContents: {text_content}"
        # 날짜 처리
        news_date = news.get('date', news.get('question_date', news.get('publish_date', '20000000'))).replace('/', '').replace('-', '')[:8] # 8자리 숫자 처리
        # url 처리
        url = news.get('url', '')
        # 고유한 docID 생성
        doc_id = news.get('doc_id') or news.get('docID') or str(uuid4())
        # project_id 처리
        if 'projectId' in news:
            project_id = news['projectId']
        elif 'project_id' in news:
            project_id = news['project_id']
        elif url != '':
            url_main = url.split('/')[2] if '//' in url else url.split('/')[0]
            url_main = url_main.split(':')[0]  # 포트 제거
            url_main = url_main.split('?')[0]  # 쿼리 파라미터 제거
            url_main = url_main.split('#')[0]  # 해시 제거
            url_main = get_url_main(url_main)
            project_id = url_main
        else:
            project_id = "wikipedia" #기본값 설정정
        # query 처리
        query = news.get('query', news.get('question', news.get('question_sentence', '')))
        metadata = {
            "date":  news_date,
            "docID": doc_id,  # doc_id가 없으면 UUID 생성
            "url": url,
            "source": ','.join(news.get('authors', [])) if isinstance(news.get('authors', []), list) else news.get('authors', ''),
            "projectId": project_id,
            "query": query,
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        time_passed = time.time() - new_time
        print(f"Documents Added: {doc.metadata['docID']} - {doc.metadata['date']} - {doc.metadata['url']}, time passed: {time_passed:.2f} seconds")
        # debugging
        # print('문서 설정')
        # print(documents)
    return documents

def create_faiss_index(documents, embeddings, base_dir="faiss_indexes", start_date=None, end_date=None):
    """문서 리스트로부터 FAISS 인덱스 생성 (GPU 최적화)
        documents : List[Document] - 문서 리스트
        embeddings : HuggingFaceEmbeddings - 임베딩 모델
        base_dir : str - FAISS 인덱스 저장 디렉토리
        start_date : str - 시작 날짜 (YYYYMMDD 형식). 없으면 전체 문서 사용
        end_date : str - 종료 날짜 (YYYYMMDD 형식). 없으면 전체 문서 사용
    """
    if not documents:
        raise ValueError("문서 리스트가 비어 있습니다.")
    
    if "_nm" in end_date or "_nm" in start_date:
        use_metadata = False
        end_date = end_date.replace('_nm', '') if end_date else None
        start_date = start_date.replace('_nm', '') if start_date else None
    else:
        use_metadata = True

    # GPU 사용 가능 여부 확인
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"FAISS index creation using device: {device}")

    # base_dir에서 날짜별 경로 호출과 생성
    old_dir_list = os.listdir(base_dir) # 기본 경로
    old_dir_list = [d for d in old_dir_list if re.match(r"20\d{1,10}", d) or re.match(r"merged_\d+.*\d+", d)]
    print("기존 디렉토리 목록:::", old_dir_list)

    # 배치 처리로 성능 향상
    batch_size = 50 if device == 'cuda' else 10
    
    # 날짜별로 문서를 그룹화
    date_groups = {}
    for document in documents:
        date_str = document.metadata.get("date", "20000000")
        date_str = process_date_string(date_str)
        start_date = process_date_string(start_date) if start_date else None
        end_date = process_date_string(end_date) if end_date else None 
        
        # 날짜가 유효하지 않거나 범위 밖이면 건너뛰기
        if start_date and compare_earlier_date(date_str, start_date):
            continue
        if end_date and compare_earlier_date(end_date, date_str):
            continue
        # date_str 지정
        if not use_metadata:
            date_str = f"{date_str}_nm"
            start_date_nm = f"{start_date}_nm" if start_date else None
            end_date_nm = f"{end_date}_nm" if end_date else None

        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(document)

    # 날짜별로 배치 처리
    for date_str, docs in date_groups.items():
        part_vector_store_dir = f"{base_dir}/{date_str}"
        if date_str not in old_dir_list:
            os.makedirs(part_vector_store_dir, exist_ok=True)
        
        try:
            part_vector_store = FAISS.load_local(part_vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            # 인덱스가 없으면 새로 생성 (배치로)
            part_vector_store = FAISS.from_documents(docs[:batch_size], embeddings)
            remaining_docs = docs[batch_size:]
            
            # 나머지 문서들을 배치로 추가
            for i in range(0, len(remaining_docs), batch_size):
                batch = remaining_docs[i:i+batch_size]
                if batch:
                    part_vector_store.add_documents(batch)
            
            part_vector_store.save_local(part_vector_store_dir)
            continue
        
        # 중복된 ID 확인 및 제거
        existing_ids = set(part_vector_store.index_to_docstore_id.values())
        new_docs = []
        for doc in docs:
            if doc.metadata["docID"] not in existing_ids:
                new_docs.append(doc)

        # 새 문서들을 배치로 추가
        if new_docs:
            for i in range(0, len(new_docs), batch_size):
                batch = new_docs[i:i+batch_size]
                if batch:
                    part_vector_store.add_documents(batch)
            part_vector_store.save_local(part_vector_store_dir)
    

    # vector_store = FAISS.load_local(base_dir, embeddings, allow_dangerous_deserialization=True)
    vector_store = merge_faiss_indexes(base_dir, embeddings)
    if not vector_store:
        raise ValueError("FAISS 인덱스 생성에 실패했습니다. 문서가 비어있거나 잘못된 형식입니다.")
        
    return vector_store

def merge_faiss_indexes(base_dir, embeddings):
    """base_dir 하위의 모든 날짜별 FAISS 인덱스를 병합"""
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    merged_index = None

    for subdir in subdirs:
        try:
            index = FAISS.load_local(subdir, embeddings, allow_dangerous_deserialization=True)
            if merged_index is None:
                merged_index = index
            else:
                merged_index.merge_from(index)
        except Exception as e:
            print(f"{subdir} 인덱스 로드 실패: {e}")

    if merged_index is None:
        raise ValueError("병합할 인덱스가 없습니다.")
    return merged_index

def get_url_main(url):
    """
    URL에서 www, co.kr, com, org, net, kr, us 등 공용 패턴을 제거하고 메인 도메인만 반환
    예: https://news.naver.co.kr/ → naver
    """
    import re
    # 도메인만 추출
    domain = url.split('//')[-1].split('/')[0]
    domain = re.sub(r'^www\.', '', domain)
    parts = domain.split('.')
    # 복합 도메인 패턴
    multi_patterns = {'co.kr', 'ac.kr', 'or.kr', 'go.kr', 'ne.kr', 're.kr'}
    # 마지막 두개가 복합 도메인인지 확인
    if len(parts) >= 3 and '.'.join(parts[-2:]) in multi_patterns:
        main = parts[-3]
    elif len(parts) >= 2:
        main = parts[-2]
    else:
        main = parts[0]
    return main

def check_jsonls(data_1, data_2):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]

def fall_back(data_1, data_2, top_k=5):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]
        datum_1["search_result"] = [article for article in datum_1["search_result"] if "text" in article]
        nb_retrieved = len(datum_1["search_result"])
        if nb_retrieved < top_k:
            datum_1["search_result"].extend(datum_2["search_result"][:top_k-nb_retrieved])
    return data_1

def create_chunks(text, chunk_size=1000, overlap=500):
    """텍스트를 지정된 크기로 오버랩하여 청킹"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # 문장 경계에서 자르기 (가능한 경우)
        if end < len(text):
            # 마지막 완전한 문장 찾기
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cut_point = max(last_period, last_newline)
            
            if cut_point > start + chunk_size // 2:  # 최소 절반 이상의 내용이 있어야 함
                chunk = text[start:cut_point + 1]
                end = cut_point + 1
        
        chunks.append(chunk.strip())
        
        # 다음 시작점 계산 (오버랩 적용)
        if end >= len(text):
            break
        start = max(start + chunk_size - overlap, end - overlap)
    
    return chunks

def process_date_string(date_string):
    """
    날짜 문자열을 'YYYYMMDD' 형식으로 변환
    예: '2023/10/01' → '20231001'
    예: '2023-1-1' → '20230101'
    """
    from datetime import datetime
    # 현재 연도 확인
    current_year = datetime.now().year

    if not date_string:
        return '20000000'  # 기본값 설정
    # 날짜 형식 -> 기호 통일
    date_string = date_string.replace('/', '-').replace('.', '-') # 슬래시를 하이픈으로 변경
    # '-'가 없으면 분석해서 반환
    if '-' not in date_string:
        if len(date_string) == 8: return date_string  # 이미 'YYYYMMDD' 형식
        elif len(date_string) == 6: return '20' + date_string   #YYMMDD -> 20YYMMDD
        elif len(date_string) == 4: # MMDD -> YYYYMMDD
            return str(current_year) + date_string  # YYYY + MMDD
        else:
            raise ValueError(f"잘못된 날짜 형식: {date_string}")
    # 날짜 문자열 파싱
    date_string_split = date_string.split('-')
    if len(date_string_split) == 3:
        str_year, str_month, str_day = date_string_split
        if len(str_year) == 2:
            str_year = '20' + str_year
        else:
            str_year = str(str_year)
        str_month = str_month.zfill(2)  # 월을 두 자리로 맞춤
        str_day = str_day.zfill(2)  # 일을 두 자리로 맞
        return f"{str_year}{str_month}{str_day}" # YYYYMMDD 형식으로 반환
    elif len(date_string_split) == 2: # MM-DD 또는 YYYY-MM 형식 가능성
        # 앞부분이 4자리 -> YYYY-MM 형식
        if len(date_string_split[0]) == 4:
            str_year = date_string_split[0]
            str_month = date_string_split[1].zfill(2)
            return f"{str_year}{str_month}01"  # 일은 01로
        # 앞부분이 2자리면서 앞에 13 이상이면 YY-MM 형식
        elif len(date_string_split[0]) == 2 and int(date_string_split[0]) >= 13:
            str_year = '20' + date_string_split[0]
            str_month = date_string_split[1].zfill(2)
            return f"{str_year}{str_month}01"  # 일은 01로
        # 나머지 경우는 MM-DD 형식
        else:
            str_month = date_string_split[0].zfill(2)
            str_day = date_string_split[1].zfill(2)
            return f"{current_year}{str_month}{str_day}"
    else:
        raise ValueError(f"잘못된 날짜 형식: {date_string}")

# retrieve single question with DPR and GCS

NEWS_CATEGORY_LISTS = [
    "Sociology",
    "Politics",
    "Economics",
    "Culture",
    "Science",
    "Technology",
    "Health",
    "Sports",
    "Entertainment",
]

# BERT Classifier를 사용하여 뉴스 카테고리 분류
def classify_news_category(text, model, tokenizer):
    """
    주어진 텍스트를 BERT 모델을 사용하여 뉴스 카테고리로 분류합니다.
    
    :param text: 분류할 뉴스 텍스트
    :param model: BERT 모델
    :param tokenizer: BERT 토크나이저
    :return: 분류된 뉴스 카테고리
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    
    return NEWS_CATEGORY_LISTS[predicted_class] if predicted_class < len(NEWS_CATEGORY_LISTS) else "Unknown"

def retrieve_single_question(question, model, retriever, tokenizer, key, engine, top_k=10, start_date=None, end_date=None, use_metadata=True):
    """
    단일 질문에 대해 DPR과 GCS를 사용하여 검색 결과를 반환합니다.
    
    :param question: 검색할 질문 문자열
    :param model: DPR 모델
    :param retriever: DPR 리트리버
    :param key: GCS API 키
    :param engine: GCS 엔진 ID
    :param top_k: 검색 결과의 상위 K개를 반환
    :return: 검색 결과 리스트
    """
    # question이 string이 아닌 경우 별도 처리
    if not isinstance(question, str):
        new_question = question.get("query", "") # 
    
    else:
        new_question = question


    search_result = run_dpr_question(new_question, retriever, model, tokenizer)

    if not search_result:
        return []

    # GCS로부터 추가 정보 가져오기
    gcs_results = gcs_search(new_question, key, engine, top_k=top_k)
    gcs_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")
    new_gcs_results = []

    # 결과 병합 및 정리
    for article in gcs_results:
        try:
            publish_date = article.get("publish_date", "2018/12/31")
            publish_date = process_date_string(publish_date)  # 날짜 문자열을 'YYYYMMDD' 형식으로 변환
            article["text"], article["authors"], article["publish_date"] = parse_article(article["url"])
            doc_chunks = create_chunks(article["text"], chunk_size=1000, overlap=500)
            for chunk in doc_chunks:
                # start_date와 end_date를 기준으로 날짜 필터링
                if start_date and compare_earlier_date(publish_date, start_date):
                    continue
                if end_date and compare_earlier_date(end_date, publish_date):
                    continue
                # 청크가 비어있지 않은 경우에만 추가
                if chunk.strip():
                    # 청크가 너무 길면 잘라내기
                    if len(chunk) > 4000:
                        chunk = chunk[:4000] + "..."
                    # 새로운 결과 추가
                    if use_metadata:
                        new_gcs_result_obj = {
                            "doc_id": article.get("doc_id", str(uuid4())),
                            "text": chunk,
                            "publish_date": article.get("publish_date", "2018/12/31"),
                            "query": question,
                            "url": article.get("url", ""),
                            "title": article.get("title", ""),
                            "search_time": gcs_time,
                            "authors": article.get("authors", []),
                        }
                    else:
                        new_gcs_result_obj = {
                            "doc_id": article.get("doc_id", str(uuid4())),
                            "text": chunk,
                        }

                    new_gcs_results.append(new_gcs_result_obj)

        except Exception as e:
            print(f"URL 파싱 오류: {e}")
            continue
    print("modified gcs_results", new_gcs_results)
    print(len(new_gcs_results), "개 문서가 검색되었습니다.")
    search_result = new_gcs_results + search_result
    # 중복 제거
    seen_urls = set()
    unique_results = []
    for result in search_result:
        if result.get('url', None) and result['url'] not in seen_urls:
            seen_urls.add(result['url'])
            unique_results.append(result)
    
    return unique_results[:top_k]  # 상위 K개 결과 반환, top_k가 None이면 전체 반환


def compare_earlier_date(date1, date2):
    """
    두 날짜 문자열을 비교하여 date1이 date2보다 이전인지 확인합니다.
    날짜 형식은 'YYYYMMDD'입니다.
    
    :param date1: 첫 번째 날짜 문자열
    :param date2: 두 번째 날짜 문자열
    :return: date1이 date2보다 이전이면 True, 아니면 False
    """
    if type(date1) is not str or type(date2) is not str:
        return False # 날짜 형식이 문자열이 아닐 경우 비교하지 않음
    elif len(date1) != 8 or len(date2) != 8:
        return False
    elif not date1.isdigit() or not date2.isdigit():
        return False
    # 날짜가 '20000000'인 경우는 유효하지 않은 날짜로 간주
    # 따라서 이 경우는 비교하지 않음
    return date1 != "20000000" and date2 != "20000000" and int(date1) < int(date2)


# 날짜 연산 
def compute_date_difference(date1, date2):
    """
    두 날짜 문자열의 차이를 계산합니다.
    날짜 형식은 'YYYYMMDD'입니다.
    
    :param date1: 첫 번째 날짜 문자열
    :param date2: 두 번째 날짜 문자열
    :return: 날짜 차이 (일 단위)
    """
    from datetime import datetime
    date_format = "%Y%m%d"
    
    d1 = datetime.strptime(date1, date_format)
    d2 = datetime.strptime(date2, date_format)
    
    return (d2 - d1).days # 정수 출력. d2가 d1보다 크면 양수, 작으면 음수

# 날짜연산2 - date1 기준 n일 전/후 날짜 계산, 음수면 전, 양수면 후
def compute_relative_date(date1, n):
    """
    date1 기준으로 n일 전/후 날짜를 계산합니다.
    날짜 형식은 'YYYYMMDD'입니다.
    
    :param date1: 기준 날짜 문자열
    :param n: 정수 (음수면 이전, 양수면 이후)
    :return: 계산된 날짜 문자열 (YYYYMMDD 형식)
    """
    from datetime import datetime, timedelta
    date_format = "%Y%m%d"
    
    d1 = datetime.strptime(date1, date_format)
    new_date = d1 + timedelta(days=n)
    
    return new_date.strftime(date_format)  # YYYYMMDD 형식으로 반환


QA_NAME_ANALYZER = {
    # formation of realtimeqa
    "realtimeqa": {
        "id": "question_id",
        "query": "question_sentence",
        "source": "question_source",
        "date": "question_date",
        "url": "question_url",
        "answers": "" # 직접 입력하지 않고 처리
    }
}

# qa_type 분석 함수
# part_dic -> 기본 입력 -> 키 통일 id, query, source, date, url, answers
# use type -> Retriever only, Retriever with Metadata
def analyze_qa_type(part_dic, qa_name="realtimeqa", question_type="MCQ", use_type="Retriever Only"):
    res_obj = {}
    if qa_name == "realtimeqa":
        rematch_keys = QA_NAME_ANALYZER[qa_name]
        # 단순 키 변경
        if use_type == "Retriever with Metadata":
            for key, value in rematch_keys.items():
                if value != "":
                    res_obj[key] = part_dic.get(value, "")
        else: # Retriever Only
            res_obj["id"] = part_dic.get("question_id", "")
            res_obj["query"] = part_dic.get("question_sentence", "")
        
        # 답변 변경
        if question_type == "MCQ":
            # 리스트 찾기
            part_choices = part_dic.get("choices", []) # 선택지
            res_obj["choices"] = part_choices
            part_answer = part_dic.get("answer", [""])[0] # 첫 번째 답변
            if isinstance(part_answer, list):
                res_obj["answers"] = part_answer
            else:
                res_obj["answers"] = [part_answer] # 선택지 리스트
        elif question_type == "Generate":
            # 단일 답변 처리 - 
            part_answer_num = int(part_dic.get("answer", [""])[0]) # 정수 변환
            real_answer = part_dic.get("choices", [""])[part_answer_num] # 실제 답변
            if isinstance(real_answer, list):
                res_obj["answers"] = real_answer
            else:
                res_obj["answers"] = [real_answer]
    
    return res_obj


def process_openai_generate(question: str, context: str, client=None, api_key=None) -> str:
    """OpenAI Generate 모드: 자유 형태 답변 생성"""
    try:
        max_context = context

        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant. Please provide a correct and simple answer based on the provided context. A pair of question and answer will be given to comprehend how to answer the question"
            },
            {
                "role": "user", 
                "content": "Context: For the third year in a row, Hong Kong received the dubious title of priciest city in the world, according to an analysis this week by global mobility company ECA International. The company calculates the list based on several factors, including the average price of groceries, rent, utilities, public transit and the strength of the local currency. \n\nQuestion: According to a recent ranking, which is the world’s most expensive city?\n\nPlease provide a correct and brief answer(a word or a phrase):"
            },
            {
                "role": "assistant", 
                "content": f"Hong Kong"
            },
            {
                "role": "user", 
                "content": f"Context: {max_context if max_context else 'No specific context provided'}\n\nQuestion: {question}\n\nPlease provide a brief answer:"
            }
        ]

        # OpenAI 클라이언트 설정
        if client is None and api_key is not None:
            client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 더 나은 성능을 위해 gpt-4 사용
            messages=messages,
            temperature=0.4,
            max_tokens=100,  # Generate 모드는 더 긴 답변 허용
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        error_msg = f"OpenAI Generate API Error: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg

def process_openai_mcq(question: str, context: str, choices=[], client=None, api_key=None) -> str:
    """OpenAI MCQ 모드: 객관식 질문에 최적화된 답변"""
    try:
        max_context = context

        # MCQ인지 확인하고 적절한 프롬프트 생성
        if any(keyword in question.lower() for keyword in ["choices:", "a.", "b.", "c.", "d.", "0.", "1.", "2.", "3.", "4."]):
            # 명확한 객관식 질문
            system_prompt = "You are an expert at answering multiple choice questions. Analyze the context carefully and select the most accurate answer. Provide only the letter/number of the correct choice followed by a brief explanation."
            user_prompt = f"Context: {max_context if max_context else 'No specific context provided'}\n\n{question}\n\nPlease select the correct answer and provide a brief explanation:"
        else:
            # 일반 질문을 MCQ 스타일로 처리
            if type(choices) is str:
                if ';' in choices: # ;로 구별
                    choices_list = choices.split(';')
                    choices_symbols = [re.match(r'([a-zA-Z0-9])\.?\b', s.strip()).group(1) for s in choices_list]
                else:
                    choices_symbols = re.findall(r'\b([a-zA-Z0-9]\.)\b', choices)
                    choices_symbols = [s.replace('.', '') for s in choices_symbols]  # a. b. c. d. -> a b c d
            elif type(choices) is list: # 리스트 -> 0, 1, 2, 3
                choices_symbols = [str(i) for i in range(len(choices))]
                if '0' not in choices[0]:
                    choices = [f"{i}. {choice}" for i, choice in enumerate(choices)]  # a. b. c. d. 형식으로 변환
                choices = '; '.join(choices)  # 리스트를 문자열로 변환

            system_prompt = "You are an expert assistant. Based on the provided context, give a concise and precise answer. If it's a factual question, provide a specific answer number from choices Keep responses focused and to the point."
            user_prompt = f"Context: {max_context if max_context else 'No specific context provided'}\n\nQuestion: {question}\n\nChoices: {choices}\n\nProvide a concise, precise answer among {', '.join(choices_symbols)}:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # OpenAI 클라이언트 설정
        if client is None and api_key is not None:
            client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # MCQ는 더 일관된 답변을 위해 낮은 temperature
            max_tokens=10,   # MCQ는 짧은 답변
            top_p=0.8,
            frequency_penalty=0,
            presence_penalty=0
        )

        answer = response.choices[0].message.content.strip()
        # 답변에서 번호 정보만 추출
        match = re.search(r'\b([a-zA-Z0-9])\b', answer)
        if match:
            answer = match.group(1).strip()
        else:
            # 번호가 없으면 전체 답변 반환
            print("[DEBUG] No specific choice found in the answer, returning full answer.")
            return answer
        return answer

    except Exception as e:
        error_msg = f"OpenAI MCQ API Error: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        return error_msg