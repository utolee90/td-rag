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

# utils.py
from utils import (
    load_news_data, create_documents, create_faiss_index, create_chunks, process_date_string, retrieve_single_question, compute_relative_date
)
from uuid import uuid4

from evaluate import accuracy, gen_eval

# keys.py
from keys import GCS_KEY, ENGINE_KEY, OPENAI_API_KEY, MODEL_PATH, MODEL_NAMES

global_api_key = OPENAI_API_KEY  # OpenAI API 키 설정

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

    base_dir = Path("faiss_indexes")  # 임시 FAISS DB 경로 수정
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

# FAISS DB에 넣기
def process_uploaded_files(files, use_type='retrieve'):
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

    total_docs = 0
    progress_html = ""
    processed_list = [] # tlfgod

    try:
        for file_idx, file in enumerate(files, 1):

            progress_html += f"<p>Processing File {file_idx}/{len(files)}: {file.name}, Process : {use_type}</p>"
            yield processed_data, progress_html

            news_data = load_news_data(file.name)

            # news_data key analysis
            assert isinstance(news_data, list), "News data should be a list."
            assert all(isinstance(news, dict) for news in news_data), "Each news item should be a dictionary."
            print(news_data[0].keys())

            # 필수 키 확인 # 다음 세 리스트의 페어 중 하나는 반드시 있어야 함
            # 기본적으로 뉴스 데이터 -> DB에 저장
            if use_type.lower() in ['retrieve', 'faiss']:
                # 기본적으로 url, title, text/content, authors, publish_date
                if "search_result" in list(news_data[0].keys()):
                    print("NEWSKEYS")     
                    assert "text" in news_data[0]["search_result"][0], "뉴스 데이터의 'search_result' 항목에 'text' 키가 있어야 합니다."
                    news_date_temp = [news["search_result"] for news in news_data]
                    news_data = [item for sublist in news_date_temp for item in sublist]  # flatten list
                    with open("news_data_temp_flattened.json", "w", encoding="utf-8") as f:
                        json.dump(news_data, f, ensure_ascii=False, indent=4)

                # 날짜별 그룹화
                date_groups = {}
                for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                    print("news keys", news.keys())
                    date = news.get('date', news.get('question_date', news.get('publish_date', '20000000')))
                    if isinstance(date, str):
                        date = process_date_string(date)
                    elif isinstance(date, datetime.date):
                        date = date.strftime("%Y%m%d")
                    else:
                        date = '20000000'  # 기본값 설정
                    if date not in date_groups:
                        date_groups[date] = []
                    date_groups[date].append(news)

                    if news_idx % max(1, len(news_data)//10) == 0:
                        progress = (news_idx + 1) / len(news_data) * 100
                        progress_html += f"<p>Progressing documents: {progress:.1f}% ({news_idx + 1}/{len(news_data)})</p>"
                        yield processed_data, progress_html

                processed_data["Index_by_date"][file.name] = {
                    "Number_of_documents": len(news_data),
                    "Number_of_documents_by_date": {date: len(articles) for date, articles in date_groups.items()}
                }
                total_docs += len(news_data)

                progress_html += f"<p>Creating Indexes... ({len(date_groups)})</p>"
                yield processed_data, progress_html

                for date_idx, (date, articles) in enumerate(date_groups.items(), 1):
                    progress_html += f"<p>Making {date} date index... ({date_idx}/{len(date_groups)})</p>"
                    yield processed_data, progress_html

                    # articles -> 
                    """{'question_id': '20250516_0_nota', 'question_date': '2025/05/16', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2025/05/us/cnn-5-things-news-quiz-may-16-sec/', 'question_sentence': 'The Qatari royal family has offered President Donald Trump which of the following expensive gifts?', 'choices': ['An armored Rolls-Royce', 'A luxury jet to replace Air Force One', 'A penthouse apartment in Doha', 'The Millennium Star diamond'], 'answer': ['1'], 'evidence': '<p class="_question-block_answer-response__copy">Trump said the Defense Department <a target="_blank" href="https://www.cnn.com/2025/05/11/politics/trump-luxury-jet-qatar-air-force-one">plans to accept the Boeing 747-8 jet</a> and retrofit it to be used as Air Force One, which raises substantial ethical and legal questions.</p>'}"""
                    article_objs = [] # 각 문서에서 document 작성 -> question_sentence -> search_result
                    for article in tqdm(articles, desc=f"Processing articles for date {date}"):
                        print(f"[DEBUG] Articles for date {date}: {article}")  # 첫 번째 기사만 출력
                        query = article.get('question_sentence', article.get('text', ''))
                        if not query:
                            continue
                        # find search_result from query by retrieve_single_question
                        top_k = 5
                        # 전역 모델 사용
                        global global_retriever, global_retriever_model, global_tokenizer
                        
                        if global_retriever is None or global_retriever_model is None or global_tokenizer is None:
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            global_retriever, global_retriever_model, global_tokenizer = load_model(
                                BASE_RETRIEVER_MODEL, 
                                top_k=top_k,
                                device=device
                            )
                        
                        end_date = article.get('question_date', article.get('date', '20000000'))
                        if isinstance(end_date, str):
                            end_date = process_date_string(end_date)
                        elif isinstance(end_date, datetime.date):
                            end_date = end_date.strftime("%Y%m%d")
                        else:
                            end_date = '20000000'

                        search_result = retrieve_single_question(
                            query, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY, 
                            top_k=top_k, start_date=None, end_date=end_date
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
            
            # realtimeqa 프로세스 처리
            elif use_type.lower() in ['qa', 'realtimeqa', 'realtime', 'cnnqa', 'newsqa']:
                answers = [] # 답변 목록
                scores = []
                answer_objs = [] # 전체 목록
                
                progress_html += f"<p>Starting QA processing for {len(news_data)} questions...</p>"
                yield processed_data, progress_html
                
                res_text = ""
                for news_idx, news in enumerate(tqdm(news_data, desc="Processing news data")):
                    res_obj = dict()  # 각 뉴스에 대한 결과 객체 초기화
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
                    print("GLOBAL GENERATOR MODEL", global_generator_model)
                    answer = search_interface.search_news(
                        global_generator_model,
                        modified_query, 
                        top_k=top_k,
                        date_info="{}/{}".format(start_date, end_date),
                        pos=False
                    )
                    score = news.get('score', 0.0)
                    answers.append(answer)
                    scores.append(score)  # score가 없으면 0.0으로 설정
                    
                    res_obj["question_id"] = news.get('question_id', f"{date}_{uuid4().hex[:8]}")
                    res_obj["prediction"] = [answer]
                    res_obj["score"] = str(score)
                    answer_objs.append(res_obj)  # 전체 답변 객체에 추가
                    res_text += json.dumps(res_obj, ensure_ascii=False) + "\n"

                # 정확도 accuracy 사용 
                try:
                    print(f"Evaluating {len(answer_objs)} answers against {len(news_data)} news data...")
                    print(answer_objs[:5])  # 디버깅용 출력
                    print(news_data[:5])  # 디버깅용 출력
                    if global_generator_model.lower() == "openai mcq":
                        eval_results = accuracy(answer_objs, news_data) # 정확도 선다형
                    elif global_generator_model.lower() == "openai":
                        eval_results = gen_eval(answer_objs, news_data) # 정확도 선택형
                    
                    # 결과를 문자열로 변환하여 저장
                    if isinstance(eval_results, dict):
                        # 딕셔너리인 경우 주요 정보만 추출
                        accuracy_score = eval_results.get('accuracy', 'N/A')
                        total_questions = eval_results.get('total', len(news_data))
                        correct_answers = eval(accuracy_score) * int(total_questions) if isinstance(accuracy_score, str) else accuracy_score * int(total_questions)
                        # correct_answers = eval_results.get('correct', 'N/A')
                        result_str = f"File: {file.name} - Accuracy: {accuracy_score}, Correct: {correct_answers}/{total_questions}"
                    else:
                        result_str = f"File: {file.name} - Result: {str(eval_results)}"
                    
                    processed_list.append(result_str)
                    print(f"✅ Evaluation completed for {file.name}: {result_str}")

                    if global_generator_model.lower() == "openai mcq":
                        accuracy_report_file = f"results/accuracy_report_{file.name.split('/')[-1]}"
                    elif global_generator_model.lower() == "openai":
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
        if use_type.lower() in ['qa', 'realtimeqa', 'realtime', 'cnnqa', 'newsqa']:
            status_msg += f"\n{len(processed_list)} evaluations completed."
            if processed_list:
                joined_list = '\n'.join(processed_list)
                status_msg += f"\nEvaluation Results:\n{joined_list}"

        progress_html += f"<p style='color: green;'>{status_msg}</p>"

        # 모든 경우에 yield로 반환 (일관성 유지)
        yield processed_data, progress_html


    except Exception as e:
        error_msg = f"? 처리 중 오류 발생: {str(e)}"
        progress_html += f"<p style='color: red;'>{error_msg}</p>"
        yield None, progress_html

def process_uploaded_files2(files):
    """QA 파일 업로드 처리 함수 (Generator 버전)"""
    # generator 형태로 결과를 yield
    for result in process_uploaded_files(files, use_type='qa'):
        yield result

def make_accuracy_reports(pred_data, gold_data, file_name="results/accuracy_report.jsonl"):
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
                res_obj["em"] = exact_match_score(pred, golds[0])
                res_obj["f1"] = f1_score(pred, golds[0])
        accuracy_results.append(res_obj)
    
    accuracy_objs = [json.dumps(res, ensure_ascii=False) for res in accuracy_results]

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(accuracy_objs))
    
    print(f"Accuracy report saved to {file_name}")
    return accuracy_results

def create_gradio_interface():
    global global_manager, global_api_key
    
    # OpenAI API 키 초기화
    global_api_key = OPENAI_API_KEY
    
    global_manager = initialize_manager()
    search_interface = SearchInterface(global_manager)
    # 최대 길이 지정
    max_length = 4000  # 최대 길이 설정 (토큰 수에 따라

    with gr.Blocks(title="Search News Systems") as demo:
        gr.Markdown("""## Search News Systems""")
        
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

        with gr.Row():
            model_type = gr.Radio(
                choices=["Local HuggingFace - Generate", "Local HuggingFace - MCQ", "OpenAI", "OpenAI MCQ"],
                label="Select Model Type",
                value="OpenAI MCQ",
            )
            
        # 모델 타입 설명
        with gr.Row():
            gr.Markdown("""
            **Model Type 설명:**
            - **Local HuggingFace - Generate**: 자세한 설명형 답변 생성
            - **Local HuggingFace - MCQ**: 객관식/단답형 질문에 최적화
            - **OpenAI**: GPT를 사용한 상세한 답변 생성
            - **OpenAI MCQ**: GPT를 사용한 객관식/정확한 답변
            """)

        with gr.Group(visible=False) as local_settings:
            with gr.Row():
                model_path = gr.Textbox(
                    value=MODEL_PATH, label="Model Path"
                )
                model_name = gr.Dropdown(
                    
                    choices= MODEL_NAMES,
                    label="Select Model",
                    value="Qwen/Qwen2.5-7B-Instruct",
                )

        with gr.Group(visible=True) as openai_settings:
            api_key = gr.Textbox(label="OpenAI API Key", type="password")

        model_status = gr.Textbox(label="Model status", interactive=False)
        init_model_btn = gr.Button("Initialize Model")


    # 검색 섹션
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Please enter your query",
                    placeholder="Example : What is the cause of the fire in the mixed-use building on December 31, 2023?",
                    elem_classes=["submit-on-enter"],
                    autofocus=True
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=25,
                    value=5,
                    step=1,
                    label="number of results"
                )
                date_info = gr.Textbox(
                    label="Date information",
                    placeholder="Example : 20231230 / Recent 30 days / 2023-01-01,2023-01-31",
                    lines=1
                )
                with gr.Column():
                    search_button = gr.Button("Search", variant="primary")
                # 검색 옵션 추가
                with gr.Column():
                    retrieve_button = gr.Button("Retrieve from query") # 검색 추출 버튼 추가`
                # 검색 옵션 추가
                with gr.Column():
                    retriever_type = gr.Radio(
                        choices=["GCS+DPR", "Contriever+Wikipedia", "DPR+Wikipedia"],
                        label="Select Retriever Type",
                        value="GCS+DPR",  # 기본값으로 GCS 선택
                    )


                warning_output = gr.HTML()

        gr.Examples(
            examples=[
                ["What is the name of the president of South Korea?", 5, "20251231/20251231"],
                 ["What is the name of the president of South Korea?", 5, "20220601/20241231"],
                  ["What is the name of the president of South Korea?", 5, "20250501/20250630"]
            ],
            inputs=[query_input, top_k, date_info],
        )
        # 답변 섹션
        gr.Markdown("### Answer")
        with gr.Row():
            results_answer = gr.Textbox(
                label="Search Result",
                lines=2,
                show_copy_button=True
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
            
            with gr.Column():
                file_output = gr.JSON(label="Current File Status")
                upload_button = gr.File(
                    label="Upload JSON/JSONL files for FAISS ",
                    file_types=[".json", ".jsonl"],
                    file_count="multiple",
                )
                status_output = gr.HTML(label="Status Output")
        
        # QA 답변 문제
        gr.Markdown("### QA Answer Retrieval")
        with gr.Row():
            
            with gr.Column():
                file_output2 = gr.JSON(label="Current File Status")
                upload_button2 = gr.File(
                    label="Upload QA JSON/JSONL files",
                    file_types=[".json", ".jsonl"],
                    file_count="multiple",
                )
                status_output2 = gr.HTML(label="Status Output")

        # 모델 설정 탭
        def update_index_info():
            return global_manager.load_created_indexes()
        # Enter key submission handler


        # 모델 변경 이벤트 연결
        def toggle_model_settings(choice):
            if choice in ["Local HuggingFace - Generate", "Local HuggingFace - MCQ"]:
                return gr.Group(visible=True), gr.Group(visible=False)
            elif choice in ["OpenAI", "OpenAI MCQ"]:
                return gr.Group(visible=False), gr.Group(visible=True)
            else:
                return gr.Group(visible=False), gr.Group(visible=True)

        # 모델 설정 변경 이벤트 연결
        model_type.change(
            toggle_model_settings,
            inputs=[model_type],
            outputs=[local_settings, openai_settings],
        )

        def init_model(model_type, path, name, api_key=None):
            try:
                if model_type == "Local HuggingFace - Generate":
                    return search_interface.init_local_model_generate(path, name)
                elif model_type == "Local HuggingFace - MCQ":
                    return search_interface.init_local_model_mcq(path, name)
                elif model_type in ["OpenAI", "OpenAI MCQ"]:
                    # OpenAI API 키를 keys.py에서 가져옴
                    result = search_interface.init_openai_model(api_key)
                    if model_type == "OpenAI MCQ":
                        return f"{result} (MCQ 모드 활성화)"
                    return result
                else:
                    return search_interface.init_openai_model(api_key)
                global_api_key = api_key  # OpenAI API 키 설정
            except Exception as e:
                return f"An error occurred during loading: {str(e)}"

        # 모델 초기화 버튼 클릭 이벤트 연결
        init_model_btn.click(
            fn=init_model,
            inputs=[model_type, model_path, model_name, api_key],
            outputs=[model_status],
        )

        # 검색 2개 결과 출력 (성능 개선)
        def double_search(model_type, query_input, top_k, date_info):
            import time
            start_time = time.time()
            
            # 답변만 추출
            answer = search_interface.search_news(model_type, query_input, top_k, date_info, False)
            # 검색결과 설명
            search_results = search_interface.search_news(model_type, query_input, top_k, date_info, True)
            # 저장 결과는 비우기
            
            processing_time = time.time() - start_time
            warning_output = f"<p style='color: green;'>✅ Process done - Used time: {processing_time:.2f} sec</p>"
            
            # list 형식으로 출력
            return answer, search_results, warning_output

        # 검색 버튼 클릭 이벤트 연결
        search_button.click(
            fn=double_search,
            inputs=[model_type, query_input, top_k, date_info],
            outputs=[results_answer, results_output, warning_output],
            show_progress=True
        )

        # 개별 쿼리 -> 검색자 추출 (최적화됨)
        def run_retrieve(query_input, top_k=10, skip_printing=False, start_date=None, end_date=None):
            import time
            
            from keys import GCS_KEY, ENGINE_KEY
            
            if not query_input.strip():
                return "<p style='color: red;'>⚠️ Warning : no query input</p>"
            
            print("query_input", query_input)
            start_time = time.time()
            
            # 전역 모델 사용 (매번 로드하지 않음)
            global global_retriever, global_retriever_model, global_tokenizer
            
            if global_retriever is None or global_retriever_model is None or global_tokenizer is None:
                print("Global models not loaded, loading now...")
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                global_retriever, global_retriever_model, global_tokenizer = load_model(
                    BASE_RETRIEVER_MODEL, 
                    top_k=top_k,
                    device=device
                )
                print("Models loaded successfully")

            retrieval_start = time.time()
            search_result = retrieve_single_question(
                query_input, global_retriever_model, global_retriever, global_tokenizer, GCS_KEY, ENGINE_KEY, 
                top_k=top_k, start_date=start_date, end_date=end_date
            )
            retrieval_time = time.time() - retrieval_start
            gcs_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")

            question_id = "userinput" + gcs_time.replace("/", "").replace(":", "")

            output = {"question_id": question_id, "query": query_input, "search_time": gcs_time, "search_result": search_result}


            if len(search_result) > 0:
                
                try: 
                    #with open("retriever_output.json", "w") as f:
                    #    json.dump(search_result, f, ensure_ascii=False, indent=4)
                    # 인덱스 생성
                    index_start = time.time()
                    docs = create_documents(output, max_length=max_length)  # max_length 인자 추가
                    # debugging
                    print("docs", docs)
                    # FAISS 인덱스 생성
                    vector_store = create_faiss_index(
                        docs, global_manager.embeddings, global_manager.base_dir, start_date=start_date, end_date=end_date
                    )
                    index_time = time.time() - index_start
                    total_time = time.time() - start_time
                    
                    warning_output = f"<p style='color: green;'>✅ 처리 완료 - 검색: {retrieval_time:.2f}초, 인덱싱: {index_time:.2f}초, 총 시간: {total_time:.2f}초</p>"
                except Exception as e:
                    warning_output = f"Retrieving Error : <p style='color: red;'>⚠️ Warning: {str(e)}</p>"
            else:
                warning_output = "Searching Error <p style='color: red;'>⚠️ Warning: no search result</p>"
            
            # skip_printing -> 텍스트 출력 대신 vector_store 반환
            if skip_printing:
                return vector_store
            else:
                return warning_output
        
        def run_with_warning(query_input):
            if not query_input.strip():
                return "<p style='color: red;'>⚠️ Warning : no query input</p>"
            return f"<p style='color: green;'>✅ Save data for '{query_input}'</p>"
        
        # 초기 경고 메시지 표시 함수
        def show_initial_warning():
            return "<p style='color: orange;'>⚠️ Warning: Retrieving process will start soon...</p>"

        # 초기 경고 메시지 표시
        retrieve_button.click(
            fn=show_initial_warning,
            inputs=[],
            outputs=[warning_output]
        )

        # 검색자 추출 버튼 클릭 이벤트 연결
        retrieve_button.click(
            fn=run_retrieve,
            inputs=[query_input],
            outputs=[warning_output]
        )

        # 인덱스 검색 정보
        index_info_button.click(
            update_index_info,
            outputs=[index_info_output]
        )
        
        # 파일 업로드 버튼 핸들러
        upload_button.change(
            fn=process_uploaded_files,
            inputs=[upload_button],
            outputs=[file_output, status_output],
            show_progress=True
        )

        # qa 파일 업로드 버튼 핸들러
        upload_button2.change(
            fn=process_uploaded_files2,
            inputs=[upload_button2],  # 'qa'로 설정하여 QA 프로세스 사용
            outputs=[file_output2, status_output2],
            show_progress=True
        )

        def update_model_type(selected_model_type):
            global global_generator_model, global_retriever_model, global_retriever, global_tokenizer

            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                global_generator_model = selected_model_type
                global_retriever, global_retriever_model, global_tokenizer = load_model(
                   'Facebook/rag-sequence-nq',
                    top_k=25,  # 최대값으로 설정하여 재로드 방지
                    device=device
                )
                
                print(f"Model updated to: {selected_model_type}")
                return f"Model successfully updated to: {selected_model_type}"
            except Exception as e:
                print(f"Error updating model: {e}")
                return f"Error updating model: {e}"

        # Connect the model_type radio button to the update_model_type function
        model_type.change(
            fn=update_model_type,
            inputs=[model_type],
            outputs=[model_status]
        )

        def initialize_global_model(choice):
            global global_retriever_model, global_retriever, global_tokenizer
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if choice == "Local HuggingFace - Generate":
                    global_retriever, global_retriever_model, global_tokenizer = load_model(
                        BASE_RETRIEVER_MODEL, 
                        top_k=25,
                        device=device
                    )
                    return "Local HuggingFace Generate model initialized."
                elif choice == "Local HuggingFace - MCQ":
                    global_retriever, global_retriever_model, global_tokenizer = load_model(
                        BASE_RETRIEVER_MODEL, 
                        top_k=25,
                        device=device
                    )
                    return "Local HuggingFace MCQ model initialized."
                elif choice in ["OpenAI", "OpenAI MCQ"]:
                    global_retriever_model = None  # OpenAI 모델은 별도 초기화 필요
                    return "OpenAI model selected. Please initialize with API key."
                else:
                    return "Invalid model type selected."
            except Exception as e:
                return f"Error initializing model: {str(e)}"

        model_type.change(
            fn=initialize_global_model,
            inputs=[model_type],
            outputs=[model_status]
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
    server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    print(f"Starting server on port {server_port}...")
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # 공유 링크 -
        server_name="0.0.0.0",  # 모든 IP에서 접근
        server_port=server_port,  # 포트 설정
        debug=True,  # 디버그 모드 활성화
    )

    # 실행 방법 CUDA_VISIBLE_DEVICES=8 python run.py (7861))