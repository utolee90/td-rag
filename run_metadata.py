import os
os.environ["HF_HOME"] = "/mnt/nvme02/home/tdrag/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nvme02/home/tdrag/.cache/huggingface"

from transformers.utils import move_cache
move_cache()

from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr
import logging, os, re
from pathlib import Path
import datetime
import warnings
import json
import torch

from utils import analyze_qa_type

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CUDA 최적화 설정
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# GPU 메모리 관리 설정
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB 사용 가능")

BASE_RETRIEVER_MODEL = "Facebook/rag-sequence-nq" # basic retriever model

logging.basicConfig(
    filename=f'vectordb_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
    )
global_manager = None
global_retriever = None
global_generator_model = "OpenAI MCQ"  # 기본 생성기 모델
global_retriever_model = "Facebook/rag-sequence-nq"  # 기본 검색자 모델
global_tokenizer = None
global_api_key = None  # OpenAI API 키를 위한 전역 변수

import pickle
from pydantic import BaseModel
from typing import Dict, Any
from tqdm import tqdm

# load data from other file
from utils import load_news_data
from manager import VectorStoreManager
from search import SearchInterface

from retrieval.dpr import run_dpr_question, load_model
from retrieval.gcs import search as gcs_search, parse_article

# langchain imports 
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import openai
import time

# utils.py
from utils import (
    load_news_data, create_documents, create_faiss_index, create_chunks, process_date_string, retrieve_single_question, compute_relative_date
)
from uuid import uuid4

from evaluate import accuracy, gen_eval

# keys.py
from keys import GCS_KEY, ENGINE_KEY, OPENAI_API_KEY, MODEL_PATH, MODEL_NAMES, EXTRACTOR_MODEL_PATH, COHERE_API_KEY

global_api_key = OPENAI_API_KEY  # OpenAI API 키 설정

global_extractor_model_path = EXTRACTOR_MODEL_PATH


log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / f'vectordb_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def initialize_manager():
    """전역 manager 및 모델 초기화"""
    global global_manager, global_retriever, global_retriever_model, global_tokenizer

    # GPU 사용 가능한지 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},  # GPU 사용 설정
        encode_kwargs={'normalize_embeddings': False}
    )

    # 검색 모델 미리 로드
    if global_retriever is None or global_retriever_model is None or global_tokenizer is None:
        print("Loading retrieval models...")
        retriever, retriever_model, tokenizer = load_model(
            BASE_RETRIEVER_MODEL, 
            top_k=25,  # 최대값으로 설정하여 재로드 방지
            device=device
        )
        global_retriever = retriever
        global_retriever_model = retriever_model
        global_tokenizer = tokenizer
        print("Retrieval models loaded successfully")

    base_dir = Path("faiss_indexes_gcs")  # 임시 FAISS DB 경로 수정
    sub_dirs = [
        d for d in base_dir.iterdir() 
        if d.is_dir() and (re.match(r"\d{4,8}", d.name) or re.match(r"merged_\d{4,8}", d.name))
    ]  # 날짜별 디렉토리 모음

    if not sub_dirs:
        print(f"No subdirectories found for {base_dir}")
    else:
        print(f"Subdirectories found: {sub_dirs}")

    global_manager = VectorStoreManager(embeddings, base_dir)

    # 모델 워밍업 실행
    warmup_models()

    return global_manager

# 검색 2개 결과 출력 (성능 개선)
def return_date_info(query_input, use_heidel_time=True, use_llm =True):
    import time
    import subprocess
    import os
    import tempfile
    import xml.etree.ElementTree as ET
    from typing import List, Dict, Optional
    start_time = time.time()

    # 결과 해석
    results_output = "Results from query:\n"
    results_output += f"Query: {query_input}\n\n"

    # heidel_time 실행
    heidel_time_dir = "/mnt/nvme02/home/tdrag/vaiv/RTRAG/heideltime"
    jar_path = os.path.join(heidel_time_dir, "target/de.unihd.dbs.heideltime.standalone.jar")
    lib_path = os.path.join(heidel_time_dir, "lib/*")
    config_path = os.path.join(heidel_time_dir, "config.props")

    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_input:
        tmp_input.write(query_input)
        tmp_input_path = tmp_input.name
    
    try:
        # Build command
        if use_heidel_time:
            cmd = [
                'java', '-cp', f"{jar_path}:{lib_path}",
                'de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone',
                tmp_input_path,
                '-c', config_path,
                '-l', "english",
                '-t', "narratives",
                '-pos', "no"
            ]

            # execute HeidelTime
            result = subprocess.run(
                cmd,
                cwd=heidel_time_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"HeidelTime failed: {result.stderr}")
            
            # Parse the output
            timeml_output = result.stdout # xml output
            root = ET.fromstring(timeml_output)
            # timexes = []
            timexes_bases = root.findall('.//TIMEX3')
            
            # 결과 해석 -> 텍스트 출력
            results_output += "### HeidelTime Results\n\n"
            for idx, val in enumerate(timexes_bases):
                results_output += f"Result {idx + 1}"
                results_output += f" - Type: {val.get('type', 'NONE')} "
                results_output += f" - Value: {val.get('value', 'NONE')} "
                results_output += f" - Text: {val.text.strip() if val.text else ''}"
                results_output += "\n"
        
    except subprocess.TimeoutExpired:
        results_output += "\n### HeidelTime Error\n\nHeidelTime execution timed out\n"
    except Exception as e:
        results_output += f"\n### HeidelTime Error\n\nError: {str(e)}\n"
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
        except Exception:
            pass  # 파일 삭제 실패해도 계속 진행

    # local llm 실행 - (일단 GPT-4o로 활용)
    try:
        if use_llm and global_api_key:

            client = openai.OpenAI(api_key=global_api_key)
            query_sample = "Today is July 13, 2025. I have a meeting tomorrow at 3 PM. Last week, I visited my grandmother."
            results_xml_output = """<?xml version="1.0"?>
    <!DOCTYPE TimeML SYSTEM "TimeML.dtd">
    <TimeML>
    <TIMEX3 tid="t5" type="DATE" value="PRESENT_REF">Today</TIMEX3> is <TIMEX3 tid="t3" type="DATE" value="2025-07-13">July 13, 2025</TIMEX3>. I have a meeting <TIMEX3 tid="t6" type="DATE" value="2025-07-14">tomorrow</TIMEX3> at <TIMEX3 tid="t7" type="TIME" value="2025-07-14T15:00">3 PM.</TIMEX3> <TIMEX3 tid="t8" type="DATE" value="2025-W28">Last week</TIMEX3>, I visited my grandmother.

    </TimeML>"""
            # 쿼리 추출 
            root = ET.fromstring(results_xml_output)
            timexes_bases = root.findall('.//TIMEX3')
            results_from_sample = ""
            for idx, val in enumerate(timexes_bases):
                results_from_sample += f"Result {idx + 1}"
                results_from_sample += f" - Type: {val.get('type', 'NONE')} "
                results_from_sample += f" - Value: {val.get('value', 'NONE')}"
                results_from_sample += f" - Text: {val.text.strip() if val.text else ''}"
                results_from_sample += "\n"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Please deduce the results from the following query: {query_sample}"},
                    {"role": "assistant", "content": f"Results: {results_from_sample}"},
                    {"role": "user", "content": f"Please deduce the results from the following query: {query_input}"},
                ],
                max_tokens=1000,
                temperature=0.2
            )

            llm_results = response.choices[0].message.content.strip()
            results_output += "\n### LLM Deduced Results\n\n"
            results_output += llm_results + "\n"

    except ET.ParseError as e:
        results_output += f"\n### Error in XML Parsing\n\nError: {str(e)}\n"
    except Exception as e:
        results_output += f"\n### Error in LLM Processing\n\nError: {str(e)}\n"

    processing_time = time.time() - start_time
    
    # text 결과 출력
    return results_output

# FAISS DB에 넣기
def process_uploaded_files(files, use_type='No Retriever with No Metadata', question_type='MCQ'):
    global global_manager, global_api_key
    if global_manager is None:
        global_manager = initialize_manager()

    if not files:
        return None, "Please upload at least one file."
    
    processed_data = {
        "Number_of_indexes": len(files),
        "Index_by_date": {},
        "Current_indices": {}
    }

    start_time = time.time()

    total_docs = 0
    progress_html = ""
    processed_list = [] # tlfgod

    # 함수 호출
    from utils import process_openai_generate, process_openai_mcq

    try:
        for file_idx, file in enumerate(files, 1):

            search_interface = SearchInterface(global_manager)
            search_interface.openai_api_key = global_api_key

            import openai
            client = openai.OpenAI(api_key=global_api_key)

            progress_html += f"<p>Processing File {file_idx}/{len(files)}: {file.name}, Process : {use_type}</p>"
            yield processed_data, progress_html

            news_data = load_news_data(file.name)

            # news_data key analysis
            assert isinstance(news_data, list), "News data should be a list."
            assert all(isinstance(news, dict) for news in news_data), "Each news item should be a dictionary."
            print(news_data[0].keys())

            # 분석 시계열 바꾸기
            analyzed_news_data = []
            for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                analyzed_news = analyze_qa_type(news, qa_name="realtimeqa", question_type=question_type, use_type=use_type)
                analyzed_news_data.append(analyzed_news)

            use_metadata = True if use_type.lower() in ['retriever with metadata', 'no retriever with metadata'] else False
            use_retriever = True if use_type.lower() in ['retriever with metadata', 'retriever with no metadata', 'retriever only'] else False

            # use_type에 따라 처리 방식 결정 - Retrieve로 시작할 때 저장
            # use_retriever가 True이면 검색 결과 저장
            search_list = [] # 각 뉴스마다 대응하기 analyzed news_data와 1:1 대응
            if use_retriever:
                search_interface.retriever = global_retriever
                search_interface.retriever_model = global_retriever_model
                search_interface.tokenizer = global_tokenizer

                for news_idx, news in enumerate(tqdm(analyzed_news_data, desc="Processing news data")):
                    print(f"[DEBUG] Analyzed news for index {news_idx}: {news}")
                    # query -> gcs_search 사용해서
                    query = news.get("query", "")
                    if not query:
                        search_list.append([])
                        continue
                    # find search_result from query by retrieve_single_question
                    top_k = 5
                    # use_metadata에 따라 날짜 설정
                    if use_metadata:
                        end_date = news.get('date', '20000101')
                        if isinstance(end_date, str):
                            end_date = process_date_string(end_date)
                        elif isinstance(end_date, datetime.date):
                            end_date = end_date.strftime("%Y%m%d")
                        else:
                            end_date = '20000101_nm'
                        if re.match(r"20[0-2][0-9][01][0-9][0-3][0-9]", end_date):
                            start_date = compute_relative_date(end_date, -30)  # 30일 전
                        else:
                            start_date = None
                        # query에서도 날짜 정보 사용해서 
                        # query_context = return_date_info(query, use_heidel_time=False, use_llm=True)
                        # query = f"Time Metadata : {query_context} is given. \nNow answer the question with given metadata  {query}" # 질문에 time_metadata 정보 삽입
                    else:
                        end_date = '20000101_nm'
                        start_date = None
                    search_result = retrieve_single_question(
                        query, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY,
                        top_k=top_k, start_date=start_date, end_date=end_date, use_metadata=use_metadata,
                        use_reranking=True,
                        rerank_method="cohere" if COHERE_API_KEY else "custom",
                        rerank_api_key=COHERE_API_KEY or OPENAI_API_KEY
                    )
                    if not search_result:
                        print(f"No search result for query: {query}")
                        search_list.append([])
                        continue
                    else:
                        print(f"Search result for query '{query}': {search_result}")
                        search_list.append(search_result)



            """
            # 각 뉴스 db에 저장 - 불필요
            if "Retriever" == use_type[0:9].strip():
                # 필수 키 확인 # 다음 세 리스트의 페어 중 하나는 반드시 있어야 함
                # 기본적으로 뉴스 데이터 -> DB에 저장
                # 데이터 가져오기 
                # key - 키 통일 id, query, source, date, url, answers
                # 날짜별 그룹화
                date_groups = {}
                for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                    print("news keys", news.keys())
                    part_date = analyze_qa_type(news, qa_name="realtimeqa", question_type=question_type, use_type=use_type)
                    date = part_date.get('date', '20000000') # 기본값 설정
                    if isinstance(date, str):
                        date = process_date_string(date)
                    elif isinstance(date, datetime.date):
                        date = date.strftime("%Y%m%d")
                    else:
                        date = '20000000' if use_metadata else '20000000_nm' # 기본값 설정
                    # date_group 키 이름 바꾸기
                    if use_metadata == False:
                        date = f"{date}_nm" # no metadata 접두어 추가
                    if date not in date_groups:
                        date_groups[date] = []
                    date_groups[date].append(part_date)

                    if news_idx % max(1, len(news_data)//10) == 0:
                        progress = (news_idx + 1) / len(news_data) * 100
                        progress_html += f"<p>Progressing documents: {progress:.1f}% ({news_idx + 1}/{len(news_data)})</p>"
                        yield processed_data, progress_html

                processed_data["Index_by_date"][file.name] = {
                    "Number_of_documents": len(news_data),
                    "Number_of_documents_by_date": {date: len(articles) for date, articles in date_groups.items()}
                }
                total_docs += len(news_data)

                # make news index
                progress_html += f"<p>Creating Indexes... ({len(date_groups)})</p>"
                yield processed_data, progress_html

                # 날짜별로 인덱싱 및 문서 처리
                for date_idx, (date, articles) in enumerate(date_groups.items(), 1):
                    progress_html += f"<p>Making {date} date index... ({date_idx}/{len(date_groups)})</p>"
                    yield processed_data, progress_html

                    # articles -> 획일화
                    {
                        'id': '20250516_0_nota', 
                        'date': '2025/05/16', 
                        'source': 'CNN', 
                        'url': 'https://edition.cnn.com/interactive/2025/05/us/cnn-5-things-news-quiz-may-16-sec/', 
                        'query': 'The Qatari royal family has offered President Donald Trump which of the following expensive gifts?', 
                        'answer': ['1']
                    }
                    article_objs = [] # 각 문서에서 document 작성 -> question_sentence -> search_result
                    for article in tqdm(articles, desc=f"Processing articles for date {date}"):
                        print(f"[DEBUG] Articles for date {date}: {article}")  # 첫 번째 기사만 출력
                        query = article.get('query', "")
                        if not query:
                            continue
                        # find search_result from query by retrieve_single_question
                        top_k = 5
                        # 전역 모델 사용해서 호출
                        global global_retriever, global_retriever_model, global_tokenizer
                        
                        if global_retriever is None or global_retriever_model is None or global_tokenizer is None:
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            global_retriever, global_retriever_model, global_tokenizer = load_model(
                                BASE_RETRIEVER_MODEL, 
                                top_k=top_k,
                                device=device
                            )
                        
                        
                        end_date = article.get('date', '20000000')
                        if isinstance(end_date, str):
                            end_date = process_date_string(end_date)
                        elif isinstance(end_date, datetime.date):
                            end_date = end_date.strftime("%Y%m%d")
                        else:
                            end_date = '20000000'

                        if not use_metadata:
                            end_date = f"{end_date}_nm"

                        search_result = retrieve_single_question(
                            query, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY, 
                            top_k=top_k, start_date=None, end_date=end_date, use_metadata=use_metadata,
                            use_reranking=True, 
                            rerank_method="cohere" if COHERE_API_KEY else "custom",
                            rerank_api_key=COHERE_API_KEY or OPENAI_API_KEY
                        )

                        if not search_result:
                            print(f"No search result for query: {query}")
                            continue
                        
                        # search_result -> question_id, search_time, search_result
                        search_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d")
                        question_id = article.get('question_id', f"{date}_{uuid4().hex[:8]}")  # question_id 생성
                        output = {"question_id": question_id, "search_time": search_time, "search_result": search_result}
                        # 문서 생성
                        article_objs.append(output)
                        

                    print(f"[DEBUG] Created document for date {date}: {article_objs}")  # 디버깅용 출력
                    # 전역 manager를 사용하여 인덱스 생성
                    global_manager.process_news_by_date({date: article_objs})

                processed_data["Current_indices"] = {
                    "Base_dir": str(global_manager.base_dir),
                    "List_of_generated_date_indexes": list(date_groups.keys())
                }
            """
            
            # 답변 처리 - 모든 경우에 해결
            # elif use_type.lower() in ['qa', 'realtimeqa', 'realtime', 'cnnqa', 'newsqa']:
            answers = [] # 답변 목록
            scores = []
            answer_objs = [] # 전체 목록
            context = "" # 우선 가져오지 않을 때는 빈문자열로 처리


            progress_html += f"<p>Starting QA processing for {len(news_data)} questions...</p>"
            yield processed_data, progress_html
            

            for news_idx, news in enumerate(tqdm(analyzed_news_data, desc="Processing news data")):
                res_obj = dict() # 답변 형식 확인하기 
                # key - id, query, answers
                # 답변 구하기
                query = news.get('query') # 
                if not query:
                    print(f"No query found for news: {news}")
                    continue

                # context 정의
                if use_retriever:
                    # 검색 결과가 있는 경우
                    search_result = search_list[news_idx]
                    if not search_result:
                        print(f"No search result for query: {query}")
                        continue
                    context = "\n".join([f"{item.get('title', 'idea')}: {item.get('text', '')}" for item in search_result])
                else:
                    # 검색 결과가 없는 경우 빈 문자열로 설정
                    if use_metadata:
                        # 메타데이터가 있으면 context를 chatgpt의 메타데이터 입력 함수 사용.
                        context = return_date_info(query, use_heidel_time=True, use_llm=True)
                    else:
                        context = ""

                res_obj['id'] = news.get('id', f"{news_idx}_{uuid4().hex[:8]}")  # id 생성
                res_obj['query'] = query
                res_obj['score'] = 0.5 # 기본 점수 설정
                # find answer from query
                if question_type.lower() == "generate":
                    # 생성형 질문
                    answer = process_openai_generate(
                        query, 
                        context, 
                        client=client
                    )
                elif question_type.lower() == "mcq":
                    # 선다형 질문
                    answer = process_openai_mcq(
                        query, 
                        context,
                        choices=news.get('choices', []),
                        client=client
                    )
                else:
                    print(f"Unsupported question type: {question_type}")
                    continue

                if not answer:
                    answer = ["0"]  # 기본값 설정
                elif isinstance(answer, str):
                    answer = [answer]
                
                res_obj['answer'] = answer
                res_obj['prediction'] = answer  # 예측 결과로 답변 사용
                answers.append(answer)
                answer_objs.append(res_obj)

                part_progress_html = progress_html + f"<p>Processed question {news_idx + 1}/{len(analyzed_news_data)}: {query}</p>"
                yield processed_data, part_progress_html

            """
            for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                res_obj = dict()  # 각 뉴스에 대한 결과 객체 초기화
                # print("news keys", news.keys())
                
                # question_ts 타임스탬프 처리
                if 'question_ts' in news:
                    ts = news['question_ts']
                    if isinstance(ts, (int, float)):
                        # UNIX 타임스탬프인 경우 (초 또는 밀리초)
                        if ts > 1e10:  # 밀리초 타임스탬프인 경우
                            ts = ts / 1000
                        date = datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d")
                    elif isinstance(ts, str):
                        # 문자열 타임스탬프인 경우
                        try:
                            # ISO 형식 시도 (예: 2024-01-01T00:00:00Z)
                            if 'T' in ts:
                                parsed_date = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                date = parsed_date.strftime("%Y%m%d")
                            # 숫자 문자열인 경우 (예: "1640995200")
                            elif ts.isdigit():
                                timestamp = float(ts)
                                if timestamp > 1e10:  # 밀리초
                                    timestamp = timestamp / 1000
                                date = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
                            else:
                                # 다른 날짜 형식 시도
                                date = process_date_string(ts)
                        except (ValueError, OSError) as e:
                            print(f"Warning: Could not parse timestamp {ts}: {e}")
                            date = process_date_string(str(ts))
                    else:
                        # 기타 형식
                        date = str(ts)[:8] if len(str(ts)) >= 8 else '20000000'
                else:
                    date = news.get('date', news.get('question_date', news.get('publish_date', '20000000')))
                    if isinstance(date, str):
                        date = process_date_string(date)
                    elif isinstance(date, datetime.date):
                        date = date.strftime("%Y%m%d")
                    else:
                        date = '20000000'
                        
                end_date = date
                # start_date = None
                start_date = compute_relative_date(end_date, -30)  # 최근 30일로 설정
                query = news.get('question_sentence', news.get('text', ''))
                # query modification

                if not query:
                    continue
                
                # 진행 상황 업데이트
                if news_idx % max(1, len(news_data)//10) == 0:
                    progress = (news_idx + 1) / len(news_data) * 100
                    progress_html += f"<p>Processing QA: {progress:.1f}% ({news_idx + 1}/{len(news_data)})</p>"
                    yield processed_data, progress_html
                    
                # find search_result from query by search_news
                top_k = 5
                # 전역 모델 사용
                search_interface = SearchInterface(global_manager)
                search_interface.openai_api_key = global_api_key # OpenAI API 키 설정
                # 선택지 수정
                choices = news.get('choices', [])
                answer_index = news.get('answer', [])
                if not choices or not answer_index:
                    print(f"No choices or answer index for query: {query}")
                    continue
                # 선택지 문자열로 변환
                formatted_choices = ", ".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
                modified_query = f"Question : {query} \nChoices : {formatted_choices} \nAnswer(the number of the correct answer):" # 선택형
                # modified_query = f"Question: {query}\n Answer(A word or phrase fitting the question):"
                
                # global_generator_model 안전 처리
                model_name = global_generator_model
                if isinstance(global_generator_model, list):
                    print(f"DEBUG: global_generator_model is a list: {global_generator_model}")
                    model_name = global_generator_model[0] if global_generator_model else "OpenAI MCQ"
                elif not isinstance(global_generator_model, str):
                    print(f"DEBUG: global_generator_model is not a string: {type(global_generator_model)}, value: {global_generator_model}")
                    model_name = str(global_generator_model)
                    
                print("GLOBAL GENERATOR MODEL", model_name)
                # date_info -> 날짜검색 OK retriever가 있으면, retriever가 없으면 검색 불가능한 날짜로 처리.
                if use_type.lower()[:len("no_retriever")].strip() == "no_retriever":
                    date_info = "20000000/20000000"  # 검색 불가능한 날짜로 설정
                else:
                    date_info = "{}/{}".format(start_date, end_date)

                answer = search_interface.search_news(
                    model_name,
                    modified_query, 
                    top_k=top_k,
                    date_info=date_info,
                    pos=False,
                    use_metadata=use_metadata,
                )
                score = news.get('score', 0.0)
                answers.append(answer)
                scores.append(score)  # score가 없으면 0.0으로 설정
                
                res_obj["question_id"] = news.get('question_id', f"{date}_{uuid4().hex[:8]}")
                res_obj["prediction"] = [answer]
                res_obj["score"] = str(score)
                answer_objs.append(res_obj)  # 전체 답변 객체에 추가
            """

            # 정확도 accuracy 사용 
            try:
                print(f"Evaluating {len(answer_objs)} answers against {len(news_data)} news data...")
                print(answer_objs[:5])  # 디버깅용 출력
                print(news_data[:5])  # 디버깅용 출력

                #pred length
                print("Length of answer_objs and news_data:", len(answer_objs), len(news_data))

                # global_generator_model이 리스트인 경우 첫 번째 요소 사용
                model_name = global_generator_model
                if isinstance(global_generator_model, list):
                    model_name = global_generator_model[0] if global_generator_model else "openai mcq"
                elif not isinstance(global_generator_model, str):
                    model_name = str(global_generator_model)
                
                if question_type == "MCQ":
                    eval_results = accuracy(answer_objs, news_data) # 정확도 선다형
                elif question_type == "Generate":
                    eval_results = gen_eval(answer_objs, news_data) # 주관식 
                
                # 결과를 문자열로 변환하여 저장
                if isinstance(eval_results, dict):
                    # 딕셔너리인 경우 주요 정보만 추출
                    total_questions = eval_results.get('total', len(news_data))
                    if question_type == "MCQ":
                        accuracy_score = eval_results.get('accuracy', 'N/A')
                        correct_answers = eval(accuracy_score) * int(total_questions) if isinstance(accuracy_score, str) else accuracy_score * int(total_questions)
                        # correct_answers = eval_results.get('correct', 'N/A')
                        result_str = f"File: {file.name} - Accuracy: {accuracy_score}, Correct: {correct_answers}/{total_questions}"
                    elif question_type == "Generate":
                        em_score = eval_results.get('em', 0)
                        f1_score = eval_results.get('f1', 0)
                        result_str = f"File: {file.name} - EM: {em_score}, F1: {f1_score}, Total: {total_questions}"
                
                else:
                    result_str = f"File: {file.name} - Result: {str(eval_results)}"
                
                processed_list.append(result_str)
                print(f"✅ Evaluation completed for {file.name}: {result_str}")

                if question_type == "MCQ":
                    accuracy_report_file = f"results/accuracy_report_{file.name.split('/')[-1]}"
                elif question_type == "Generate":
                    accuracy_report_file = f"results/accuracy_report_{file.name.split('/')[-1].replace('.json', '_gen.json')}"
                accuracy_reports = make_accuracy_reports(answer_objs, news_data, file_name=accuracy_report_file)
                print("accuracy_reports")
                for report in accuracy_reports[:5]:
                    print(report)
                
            except Exception as eval_error:
                error_str = f"File: {file.name} - Evaluation Error: {str(eval_error)}"
                processed_list.append(error_str)
                print(f"⚠️ Evaluation failed for {file.name}: {eval_error}")
                

        status_msg = f"? Processed Finished:\n {len(files)} files, {total_docs} documents processed."
        
        status_msg += f"\n{len(processed_list)} evaluations completed."
        time_spend = time.time() - start_time # 소요시간 (초로 표현)
        if processed_list:
            joined_list = '\n'.join(processed_list)
            status_msg += f"\nEvaluation Results:\n{joined_list} \nTime spent: {time_spend:.2f} seconds"

        progress_html += f"<p style='color: green;'>{status_msg}</p>"

        # 모든 경우에 yield로 반환 (일관성 유지)
        yield processed_data, progress_html


    except Exception as e:
        error_msg = f"? 처리 중 오류 발생: {str(e)}"
        progress_html += f"<p style='color: red;'>{error_msg}</p>"
        yield None, progress_html


def make_accuracy_reports(pred_data, gold_data, file_name="results/metadata_extraction.jsonl"):
    """pred_data의 정답 결과와 gold_data의 정답 결과를 비교하여 정확도 보고서를 생성하는 함수"""
    assert len(pred_data) == len(gold_data), "Prediction and gold data must have the same length."
    accuracy_results = []
    obj_format = {
        "question_id": "",
        "type": "mcq", #mcq or generate
        "prediction": [],
        "answer": [],
        "em": 0, # exact match or choice match(correct choice)
        "f1": 0, # f1 score
        "score": 0.0, # score
    }
    for pred, gold in zip(pred_data, gold_data):
        from evaluate import exact_match_score, f1_score
        import itertools
        res_obj = obj_format.copy()
        res_obj["question_id"] = pred.get("question_id", gold.get("question_id", "unknown"))
        res_obj["prediction"] = pred.get("prediction", [])
        res_obj["type"] = "mcq" if str(res_obj["prediction"][0]).isnumeric() else "generate" # 타입
        res_obj["score"] = float(pred.get("score", 0.0))  # score 값 설정
        if res_obj["type"] == "mcq":
            res_obj["answer"] = gold.get("answer", [])
            res_obj["em"] = int(res_obj["prediction"][0] in res_obj["answer"])
        else:
            answer_choices = gold.get("choices", [])
            answer_num = gold.get("answer", [])
            if not answer_choices or not answer_num:
                res_obj["answer"] = []
            else:
                pred = pred.get("prediction", [""])
                golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
                golds = [' '.join(perm) for perm in list(itertools.permutations(golds))]
                res_obj["answer"] = [answer_choices[int(num)] for num in answer_num]
                res_obj["em"] = exact_match_score(pred, golds)
                res_obj["f1"] = f1_score(pred, golds)
        accuracy_results.append(res_obj)
    
    accuracy_objs = [json.dumps(res, ensure_ascii=False) for res in accuracy_results]

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(accuracy_objs))
    
    print(f"Accuracy report saved to {file_name}")
    return accuracy_results

# 인터페이스 생성
def create_gradio_interface():
    global global_manager, global_api_key
    
    # OpenAI API 키 초기화
    global_api_key = OPENAI_API_KEY
    
    global_manager = initialize_manager()
    search_interface = SearchInterface(global_manager)
    # 최대 길이 지정
    max_length = 4000  # 최대 길이 설정 (토큰 수에 따라

    # Gradio 인터페이스 설정
    with gr.Blocks(title="Find Data from Query") as demo:
        gr.Markdown("""## Find Data from Query""")
        
        # 시스템 상태 표시
        with gr.Row():
            system_status = gr.HTML(value=f"""
                <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                    <b>System Status:</b><br>
                    💻 Device: {'🟢 GPU (CUDA)' if torch.cuda.is_available() else '🟡 CPU'}<br>
                    🧠 Models: {'🟢 Loaded' if global_retriever else '🔴 Not Loaded'}<br>
                    📚 Embeddings: {'🟢 Ready' if global_manager else '🔴 Not Ready'}
                </div>
            """)

        with gr.Group(visible=True) as openai_settings:
            api_key = gr.Textbox(label="OpenAI API Key", type="password")

        model_status = gr.Textbox(label="Model status", interactive=False)
        init_model_btn = gr.Button("Initialize Model")


        # 검색 섹션
        with gr.Row():
            with gr.Column():
                # 쿼리 검색
                query_input = gr.Textbox(
                    label="Please enter your query",
                    placeholder="Example : What is the cause of the fire in the mixed-use building on December 31, 2023?",
                    elem_classes=["submit-on-enter"],
                    autofocus=True
                )
                # top_k 슬라이더
                top_k = gr.Slider(
                    minimum=1,
                    maximum=25,
                    value=5,
                    step=1,
                    label="number of results"
                )
                # 검색 버튼
                with gr.Column():
                    search_button = gr.Button("Search", variant="primary")

                # 경고 메시지 출력
                warning_output = gr.HTML()

        gr.Examples(
            examples=[
                ["What is the year of the current president of the United States election?"]
            ],
            inputs=[query_input],
        )

        # 검색 결과 설명 섹션
        gr.Markdown("### Explanation of Search Results")
        with gr.Row():
            results_output = gr.Textbox(
                label="News Search Results",
                lines=10,
                show_copy_button=True
            )

        # 인덱스 관리 섹션
        with gr.Row():
            with gr.Column():
                index_info_button = gr.Button("Check Index Status", variant="secondary")
                index_info_output = gr.JSON(label="Index Status")
        
        # 파일 업로드 섹션
        gr.Markdown("### File Upload For Retreival and Saving for FAISS")
        with gr.Row():
            # data 
            gr.Markdown("### Select Date Range")
            date_range = gr.Textbox(label="Date Range - YYMMDD/YYMMDD")

        with gr.Row():

            with gr.Column():
                exp_type = gr.Radio(
                    choices=["No Retriever and No Metadata", "No Retriever with Metadata", "Retriever Only", "Retriever with Metadata"],
                    label="Select the type of experiment",
                    value="No Retriever and No Metadata",
                )
                question_type = gr.Radio(
                    choices=["MCQ", "Generate"],
                    label="Select the type of question",
                    value="MCQ",
                )
            
            with gr.Column():
                file_output = gr.JSON(label="Current File Status")
                upload_button = gr.File(
                    label="Upload JSON/JSONL files for FAISS ",
                    file_types=[".json", ".jsonl"],
                    file_count="multiple",
                )
                status_output = gr.HTML(label="Status Output")
        
        # 모델 설정 탭
        def update_index_info():
            return global_manager.load_created_indexes()
        # Enter key submission handler


        def init_model(api_key=None):
            global global_api_key
            try:
                result = search_interface.init_openai_model(api_key)
                global_api_key = api_key  # OpenAI API 키 설정
                return result
                
            except Exception as e:
                return f"An error occurred during loading: {str(e)}"

        
        # 검색 버튼 클릭 이벤트 연결
        search_button.click(
            fn=return_date_info,
            inputs=[query_input],
            outputs=[results_output],
            show_progress=True
        )
        
        # 모델 초기화 버튼 클릭 이벤트 연결
        init_model_btn.click(
            fn=init_model,
            inputs=[api_key],
            outputs=[model_status],
        )

        # 인덱스 검색 정보
        index_info_button.click(
            update_index_info,
            outputs=[index_info_output]
        )
        
        # 파일 업로드 버튼 핸들러
        upload_button.change(
            fn=process_uploaded_files,
            inputs=[upload_button, exp_type, question_type],
            outputs=[file_output, status_output],
            show_progress=True
        )

    return demo

def warmup_models():
    """모델 워밍업 - 첫 번째 요청 지연시간 감소"""
    global global_retriever, global_retriever_model, global_tokenizer, global_manager
    
    try:
        print("Warming up models...")
        
        # 더미 쿼리로 모델 워밍업
        dummy_query = "test query for warmup"
        
        if global_retriever and global_retriever_model and global_tokenizer:
            # 간단한 인코딩 테스트
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                # GPU 메모리 워밍업
                import time
                start_time = time.time()
                
                # 임베딩 모델 워밍업
                global_manager.embeddings.embed_query(dummy_query)
                
                print(f"Model warmup completed in {time.time() - start_time:.2f} seconds")
            
        print("✅ Models are warmed up and ready")
        
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

# ...existing code...

if __name__ == "__main__":
    import sys
    server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 7870
    print(f"Starting server on port {server_port}...")
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # 공유 링크 -
        server_name="0.0.0.0",  # 모든 IP에서 접근
        server_port=server_port,  # 포트 설정
        debug=True,  # 디버그 모드 활성화
    )

    # 실행 방법 CUDA_VISIBLE_DEVICES=8 python run.py (7861))