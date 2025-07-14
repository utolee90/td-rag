import retrieval_utils.hf_env
import torch, datetime
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer
from retrieval_utils.tools import read_jsonl, add_today
from datasets import load_dataset, load_from_disk
import os
import transformers
import pickle

from keys import MODEL_PATH

def run_dpr(in_file, out_file, model = 'Facebook/rag-sequence-nq', as_of=False):
    questions = read_jsonl(in_file)
    # print("MODEL", model)
    retriever, model, tokenizer = load_model(model)
    outputs = []
    for question in questions:
        with torch.no_grad():
            question_sentence = question["question"]
            search_result = run_dpr_question(question_sentence, retriever, model, tokenizer, as_of)
        search_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d/%H:%M")
        output = {"question_id": question["question_id"], "search_time": search_time, "search_result": search_result}
        outputs.append(output)
    return outputs

def run_dpr_question(question, retriever, model, tokenizer, as_of=False):
    sentence = question
    # lowercase by default
    sentence = sentence.lower()

    inputs = tokenizer(sentence, padding = True, return_tensors = "pt")
    input_ids = inputs["input_ids"].to(model.device)
    question_hidden_states = model.question_encoder(input_ids)[0]
    
    # retriever의 top_k 설정을 명시적으로 적용
    top_k = retriever.config.top_k
    
    # RagRetriever 호출
    try:
        docs_dict = retriever(input_ids.cpu().numpy(), question_hidden_states.detach().cpu().numpy(), return_tensors = "pt", n_docs=top_k)
    except TypeError:
        # n_docs 매개변수가 지원되지 않는 경우 기본 호출
        docs_dict = retriever(input_ids.cpu().numpy(), question_hidden_states.detach().cpu().numpy(), return_tensors = "pt")

    question_hidden_states = model.question_encoder(input_ids)[0]
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2).to(question_hidden_states.device)
    ).squeeze(1)
    doc_ids = [str(int(doc_id)) for doc_id in docs_dict["doc_ids"][0]]
    docs = tokenizer.batch_decode(docs_dict['context_input_ids'], skip_special_tokens=True)
    docs = [doc.strip().split(' // ')[0].strip() for doc in docs]
    doc_scores = [str(float(doc_score)) for doc_score in doc_scores[0]]
    
    search_result = []
    for doc_idx in range(len(docs)):
        doc_text = docs[doc_idx]
        doc_chunks = create_chunks(doc_text, chunk_size=1000, overlap=500)
        for chunk in doc_chunks:
            if as_of:
                chunk = add_today(chunk)
            search_result.append({
                "doc_id" : doc_ids[doc_idx], 
                "text" : chunk, 
                "doc_score" : doc_scores[doc_idx], 
                "publish_date" : "2018/12/31",
                "query": sentence
            })
        if len(doc_chunks) == 0:  # 만약 청크가 없다면
            search_result.append({
                "doc_id" : doc_ids[doc_idx], 
                "text" : docs[doc_idx], 
                "doc_score" : doc_scores[doc_idx], 
                "publish_date" : "2018/12/31",
                "query": sentence
            })
    
    return search_result

def check_pkl_file(index_path):
    try:
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        print("pkl 파일이 성공적으로 로드되었습니다.")
        return data
    except Exception as e:
        print(f"pkl 파일 로드 실패: {e}")
        return None

def load_model(model_name, top_k=10, device='cpu'):
    model_folder = MODEL_PATH  # ANDLab연구실 서버 경로 설정

    model_name = model_folder + model_name

    # 캐시 디렉토리를 명시적으로 설정
    cache_dir = '/mnt/nvme02/home/tdrag/.cache'

    # 데이터셋 경로
    data_dir = os.path.join(cache_dir, "wiki_dpr/psgs_w100.nq.compressed/")
    index_path = os.path.join(cache_dir, "wiki_dpr/psgs_w100.tsv.pkl")
    index_path2 = os.path.join(cache_dir, "wiki_dpr/")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")
    else:
        print("Index file found at", index_path)

    # 데이터셋 로드
    dataset = load_from_disk(data_dir)
    passages = dataset['train']['text']
    pickle_data = check_pkl_file(index_path)

    print("MODEL_NAME::", model_name)
    print(f"Loading model on device: {device}")

    # 환경 변수 설정
    os.environ["TRUST_REMOTE_CODE"] = "True"

    # retriever 로드 및 디바이스 설정
    try:
        retriever = RagRetriever.from_pretrained(
            model_name, 
            index_name="legacy", 
            passages=passages,
            index_path=index_path2,
            use_dummy_dataset=False
        )
        retriever.config.top_k = top_k

        # RagRetriever도 디바이스에 따라 설정
        if device == 'cuda' and torch.cuda.is_available():
            # retriever 내부 모델들을 GPU로 이동
            if hasattr(retriever, 'question_encoder'):
                retriever.question_encoder = retriever.question_encoder.cuda()
            if hasattr(retriever, 'ctx_encoder'):
                retriever.ctx_encoder = retriever.ctx_encoder.cuda()

        print(f"RagRetriever loaded with top_k={top_k} from {model_name} on {device}")
    except Exception as e:
        print("RagRetriever loading failed:: Exception::", e)
        raise

    try:
        # 모델 로드 및 디바이스 설정
        generation_model = RagSequenceForGeneration.from_pretrained(model_name)

        # device 매개변수에 따라 적절한 디바이스로 이동
        if device == 'cuda' and torch.cuda.is_available():
            generation_model = generation_model.cuda()
            print(f"Model moved to GPU: {torch.cuda.get_device_name()}")

            # GPU 메모리 최적화
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        elif device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, using CPU instead")
            generation_model = generation_model.cpu()
        else:
            generation_model = generation_model.cpu()
            print("Model loaded on CPU")

    except Exception as e:
        print("Generation model loading failed:: Exception::", e)
        raise

    try:
        tokenizer = RagTokenizer.from_pretrained(model_name)
    except Exception as e:
        print("Tokenizer loading failed:: Exception::", e)
        raise

    return retriever, generation_model, tokenizer


def create_chunks(text, chunk_size=1000, overlap=500):
    """텍스트를 청킹하는 유틸리티 함수"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > start + chunk_size // 2:
                chunk = text[start:cut_point + 1]
                end = cut_point + 1
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(start + chunk_size - overlap, end - overlap)
    return chunks