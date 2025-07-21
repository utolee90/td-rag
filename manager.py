import os
import logging
import faiss
import torch
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from typing import Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4
import numpy as np
import pickle, json, re
from dateutil.parser import parse
from utils import DocumentV2, MergedDataV2, SafeUnpickler, create_documents, group_news_data_by_date, join_search_data
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from keys import MODEL_PATH

class VectorStoreManager:
    def __init__(self, embeddings, base_dir='faiss_indexes'):
        self.embeddings = embeddings
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.loaded_indexes = {}  # 로드된 인덱스
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"VectorStoreManager using device: {self.device}")

    def load_created_indexes(self) -> dict:
        """기존에 생성된 모든 인덱스 정보 로드"""
        indexes_info = {
            "Number_of_indexes": 0,
            "Index_by_date": {},
            "Merged_indexes": []
        }

        try:
            # 기본 Index_by_date 검색
            date_pattern = re.compile(r'(\d{8})')
            # Merged_indexes 패턴
            merged_pattern = re.compile(r'merged_.*')
            for path in self.base_dir.iterdir():
                if path.is_dir() and (path.name.startswith("20") or path.name.startswith("merged_20")):
                    # Index_by_date 확인
                    date_match = date_pattern.match(path.name)
                    if date_match:
                        date = date_match.group(1)
                        try:
                            # 인덱스 메타데이터 확인
                            index_size = sum(1 for _ in path.glob("*.faiss"))
                            docstore_size = sum(1 for _ in path.glob("*.pkl"))

                            indexes_info["Index_by_date"][date] = {
                                "Path": str(path),
                                "Number_of_files": index_size,
                                "Storage_size": docstore_size
                            }
                        except Exception as e:
                            logging.error(f"Error checking index {date}: {e}")
                            continue

                    # Merged_indexes 확인
                    elif merged_pattern.match(path.name):
                        indexes_info["Merged_indexes"].append({
                            "Name": path.name,
                            "Path": str(path)
                        })

            indexes_info["Number_of_indexes"] = len(indexes_info["Index_by_date"])
            logging.info(f"Found {indexes_info['Number_of_indexes']} date indexes and {len(indexes_info['Merged_indexes'])} merged indexes")

            return indexes_info

        except Exception as e:
            logging.error(f"Error loading indexes: {e}")
            return indexes_info

    def load_specific_index(self, date_or_name: str):
        """특정 날짜 또는 이름의 인덱스 로드"""
        try:
            # 캐시에 있으면 바로 반환
            if date_or_name in self.loaded_indexes:
                return self.loaded_indexes[date_or_name]

            index_path = self.base_dir / f"{date_or_name}"

            if not index_path.exists():
                logging.warning(f"Index {date_or_name} not found")
                return None

            index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            # 새 인덱스 하용

            # 캐시에 저장
            self.loaded_indexes[date_or_name] = index
            logging.info(f"Successfully loaded index {date_or_name}")
            # self.loaded_indexes[date_or_name] = new_index
            # logging.info(f"Successfully loaded index {date_or_name}_new")

            return index

        except Exception as e:
            logging.error(f"Error loading index {date_or_name}: {e}")
            return None

    def create_index(self, date, documents, ids):
        """단일 날짜의 인덱스 생성"""
        index_dir = self.base_dir / f'{date}'
        index_dir.mkdir(exist_ok=True)

        dimension = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        try:
            vector_store.add_documents(documents=documents, ids=ids)
            vector_store.save_local(str(index_dir))
            logging.info(f"Created index for date {date} with {len(documents)} documents")
        except Exception as e:
            logging.error(f"Error creating index for date {date}: {e}")
            raise

    def process_news_by_date(self, news_data):
        """뉴스 데이터를 날짜별로 분류하여 처리"""
        try:
            import psutil, time
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.time()

            BATCH_SIZE = 1000 # 한 번에 처리할 문서 수
            BATCH_SIZE = 100 # 한번에 처리할 문서 수
            SAVE_INTERVAL = 1000  # 중간 저장 간격
            SAVE_INTERVAL = 100  # 중간 저장 간격

            logging.info(f"시작 메모리 사용량: {initial_memory:.2f}MB")


            # 날짜별 그룹화
            # 뉴스데이터 구조 파악하기
            if not isinstance(news_data, dict):
                raise ValueError("news_data는 딕셔너리 형식이어야 합니다.")
            #if not all(isinstance(news, list) for news in news_data):
            #    raise ValueError("news_data의 모든 항목은 리스트트 형식이어야 합니다.")
            
            

            # 뉴스 데이터 딕셔너리 출력

            total_processed = 0
            print("STRUCT", news_data.keys())
            if news_data.get('seach_result', None) is not None:
                news_data = join_search_data(news_data)  # 뉴스 데이터 병합
            print(f"Grouped news data by date: {news_data.keys()}")
            
            for date, articles in tqdm(news_data.items()):
                print(f"Find Articles for date: {date}, total articles: {len(articles)}, article_sample:{articles[0]}")
                index_name = f"{date}"
                #if (self.base_dir / index_name).exists():
                #    logging.info(f"Index for date {date} already exists")
                #    continue
                logging.info(f"Creating index for date {date} ({len(articles)} articles)")
                documents = create_documents(articles)
                print("documents:::", documents, "\n")
                ids = [str(uuid4()) for _ in range(len(documents))]
                self.create_index(date, documents, ids)  # ✅ create_index 호출 추가
                logging.info(f"Index created for date {date} with {len(documents)} documents")

                # 배치 처리
                vector_store = None  # ✅ 변수 미리 정의
                
                for i in range(0, len(articles), BATCH_SIZE):
                    batch = articles[i:i + BATCH_SIZE]
                    documents = create_documents(batch)
                    uuids = [str(uuid4()) for _ in range(len(documents))]
                    print("documents:::", documents, "\n")

                    try:
                        # 인덱스 생성 최적화
                        if i == 0:  # 첫 배치에서만 인덱스 초기화
                            dimension = len(self.embeddings.embed_query("hello world"))
                            
                            # GPU 사용 가능한지 확인하고 적절한 인덱스 타입 선택
                            if self.device == 'cuda' and torch.cuda.is_available():
                                try:
                                    # GPU용 인덱스 (더 간단한 FlatL2 사용)
                                    index = faiss.IndexFlatL2(dimension)
                                    
                                    # FAISS GPU 지원 확인
                                    if hasattr(faiss, 'StandardGpuResources') and torch.cuda.device_count() > 0:
                                        # GPU로 인덱스 이동
                                        res = faiss.StandardGpuResources()
                                        index = faiss.index_cpu_to_gpu(res, 0, index)
                                        print(f"✅ FAISS GPU 인덱스 생성 완료 (device: {self.device})")
                                    else:
                                        print("⚠️ FAISS GPU 지원 없음, CPU 인덱스 사용")
                                        
                                except Exception as gpu_error:
                                    print(f"⚠️ GPU 인덱스 생성 실패, CPU로 대체: {gpu_error}")
                                    index = faiss.IndexFlatL2(dimension)
                            else:
                                # CPU용 인덱스 (nlist 계산 개선)
                                quantizer = faiss.IndexFlatL2(dimension)
                                # 문서 수에 따라 적절한 nlist 계산
                                nlist = min(100, max(10, len(documents) // 4))
                                
                                try:
                                    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                                    
                                    # 학습 데이터 준비 (첫 배치 사용)
                                    doc_texts = [doc.page_content for doc in documents if doc.page_content.strip()]
                                    if not doc_texts:
                                        raise ValueError("No valid document texts for training")
                                        
                                    embeddings_matrix = self.embeddings.embed_documents(doc_texts)
                                    embeddings_np = np.array(embeddings_matrix, dtype=np.float32)
                                    
                                    # 임베딩 차원 확인
                                    if embeddings_np.shape[1] != dimension:
                                        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {embeddings_np.shape[1]}")
                                    
                                    # 학습 데이터가 충분한지 확인
                                    if len(embeddings_np) >= nlist:
                                        index.train(embeddings_np)
                                        print(f"✅ IVFFlat 인덱스 학습 완료 (nlist={nlist}, samples={len(embeddings_np)})")
                                    else:
                                        # 학습 데이터가 부족하면 FlatL2 사용
                                        print(f"⚠️ 학습 데이터 부족 ({len(embeddings_np)} < {nlist}), FlatL2 인덱스 사용")
                                        index = faiss.IndexFlatL2(dimension)
                                        
                                except Exception as faiss_error:
                                    print(f"⚠️ IVFFlat 인덱스 생성 실패, FlatL2로 대체: {faiss_error}")
                                    index = faiss.IndexFlatL2(dimension)

                            vector_store = FAISS(
                                embedding_function=self.embeddings,
                                index=index,
                                docstore=InMemoryDocstore(),
                                index_to_docstore_id={}
                            )

                        if vector_store is not None:
                            vector_store.add_documents(documents=documents, ids=uuids)
                            total_processed += len(batch)

                        # 진행 상황 및 메모리 사용량 로깅
                        current_memory = process.memory_info().rss / 1024 / 1024
                        elapsed_time = time.time() - start_time
                        docs_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0

                        logging.info(f"""
                            진행 상황:
                            - 처리된 문서: {total_processed}
                            - 현재 메모리: {current_memory:.2f}MB (증가: {current_memory - initial_memory:.2f}MB)
                            - 처리 속도: {docs_per_second:.2f} 문서/초
                        """)

                        # 중간 저장
                        if total_processed % SAVE_INTERVAL == 0 and vector_store is not None:
                            temp_path = self.base_dir / f"temp_{index_name}"
                            vector_store.save_local(str(temp_path))
                            logging.info(f"중간 저장 완료: {total_processed} 문서")

                    except Exception as e:
                        logging.error(f"Error processing batch for date {date}: {e}")
                        continue

                # 최종 저장
                if vector_store is not None:
                    vector_store.save_local(str(self.base_dir / index_name))
                    logging.info(f"Created index for date {date} with {len(articles)} documents")
                else:
                    logging.error(f"Failed to create vector store for date {date}")

            final_memory = process.memory_info().rss / 1024 / 1024
            total_time = time.time() - start_time
            logging.info(f"""
                처리 완료:
                - 총 처리 문서: {total_processed}
                - 최종 메모리: {final_memory:.2f}MB (총 증가: {final_memory - initial_memory:.2f}MB)
                - 평균 처리 속도: {total_processed / total_time:.2f} 문서/초
                - 총 소요 시간: {total_time:.2f}초
            """)

        except Exception as e:
            logging.error(f"Error in process_news_by_date: {e}")
            logging.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # 부분적으로라도 생성된 인덱스가 있다면 저장 시도
            try:
                if 'vector_store' in locals() and vector_store is not None:
                    emergency_path = self.base_dir / "emergency_save"
                    vector_store.save_local(str(emergency_path))
                    logging.info(f"Emergency save completed to {emergency_path}")
            except Exception as save_error:
                logging.error(f"Emergency save failed: {save_error}")
            
            raise

    def extract_dates_from_query(self, query):
        """쿼리에서 날짜 관련 정보 추출"""
        dates = []
        current_date = datetime.now()

        # 현재 날짜 표현 처리
        today_patterns = [r'현재', r'지금', r'오늘']
        for pattern in today_patterns:
            if re.search(pattern, query):
                dates.append(current_date.strftime("%Y%m%d"))

        # 직접적인 날짜 패턴
        date_patterns = [
            r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{4}\s*년?\s*\d{1,2}\s*월?\s*\d{1,2}\s*일?)'
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                try:
                    date_str = match.group(1)
                    parsed_date = parse(date_str, fuzzy=True)
                    dates.append(parsed_date.strftime("%Y%m%d"))
                except:
                    continue

        return sorted(list(set(dates)))

    def merge_date_range(self, start_date, end_date, output_name=None, use_metadata=True):
        """날짜 범위의 인덱스들을 병합"""
        if output_name is None:
            output_name = f"merged_{start_date}_to_{end_date}"

        merged_store = None
        date_range = self._date_range(start_date, end_date)

        for date in date_range:
            # use_metadata에 따라 인덱스 경로 결정
            if use_metadata:
                index_path = self.base_dir / f"{date}"
            else:
                # 메타데이터가 없는 경우 _nm 접미사 확인
                if not date.endswith("_nm"):
                    date = f"{date}_nm"
                index_path = self.base_dir / f"{date}"
                
            if not index_path.exists() or not any(index_path.glob("*.faiss")): # faiss 파일이 없으면 건너뛰기
                continue
            
            print("index_path:::", index_path) # 실행되는지 확인.
            # pydantic.BaseModel = pydantic.v1.BaseModel
            pickle_file_name = "index.pkl"
            fixed_file_name = "index_fixed.pkl"

            # ✅ 기존 인덱스 로드
            with open(index_path / pickle_file_name, 'rb' ) as f:
                try:
                    unpickler = SafeUnpickler(f)
                    pkl_data = unpickler.load()
                    print("FAISS load successfully")
                    # print(len(current_store), type(current_store[0]), type(current_store[1]))
                    # with open('write.txt', 'w') as g:
                    #    g.write(str(current_store))
                except Exception as e:
                    print("PICKLE_ERR:::", e)


            # ✅ Pydantic v1 → v2 변환 적용
            converted_data = pkl_data[0].__dict__
            pkl_data_v2 = DocumentV2.from_v1(converted_data)
            id_mapping = pkl_data[1]
            merged_data = MergedDataV2(document=pkl_data_v2, id_mapping=id_mapping)


            # ✅ 변환된 데이터를 다시 저장
            with open(fixed_file_name, "wb") as f_fixed:
                pickle.dump(merged_data, f_fixed)

            
            current_store = FAISS.load_local(
                str(index_path), 
                self.embeddings,  
                allow_dangerous_deserialization=True
            )
            

            if merged_store is None:
                merged_store = current_store
                #merged_store = new_store
            else:
                merged_store.merge_from(current_store)
                # merged_store = merged_store.merge_from(new_store)

        if merged_store:
            output_path = self.base_dir / f"{output_name}"
            output_path.mkdir(exist_ok=True)
            merged_store.save_local(str(output_path))

        return merged_store

    def _date_range(self, start_date, end_date):
        """날짜 범위 생성"""
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

        return date_list

    def search_with_date(self, query, k=5, date_info=None, use_metadata=True):
        """날짜 정보를 고려한 검색 실행"""

        # 날짜 없을 때 기본 날짜 범위는 최근 1개월
        if not date_info:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            print("No date_info provided, using default range:", start_date, end_date)
            # start_date, end_date =  "20181231", "20181231"
        else:
            # 날짜가 있으면 텍스트 정보를 통해 날짜 추출
            dates_list = self.extract_dates_from_query(date_info)
            # 텍스트로 날짜 추출 가능한 경우
            if len(dates_list) >= 1:
                start_date, end_date = dates_list[0], dates_list[-1]
            # 추출 불가능한 경우 LLaMA-kr 모델을 통해 날짜 추출 시도
            else:
                model_path = MODEL_PATH + 'Beomi/Llama-3-Open-Ko-8B'
                tokenizer = AutoTokenizer.from_pretrained(model_path , use_fast=False, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
                model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
                # 모델을 통해 날짜 추출
                query_ext_dates = f"주어진 문장이 있을 떄 시작일과 끝일을 YYYYMMDD 형식의 8자리 숫자로 변환해서 추출해봅시다.\n" \
                                f"예를 들면 '23년 12월 11일부터 30일까지'라는 문장이 주어지면 답은 시작일인 2023/12/11 끝일인 2023/12/30임을 이해하고 '20231211,20231230'으로 추출해야 합니다. \n" \
                                f"또한 '최근 한달'라는 입력이 들어오면 시스템이 입력한 기준의 날짜를 자동으로 계산한 뒤에 한 달 전의 시점의 날짜를 입력합니다. 입력일 기준 2025년 5월 30일일 때는 답을 '20250501,20250530' 형식으로 추출해야 합니다.\n" \
                                f"마지막으로 '2024년 1년동안'이라는 입력이 들어오면 시작일은 2024/01/01, 끝일은 2024/12/31로 이해하고 답을 '20240101,20241231'형식으로 입력해야 합니다.\n" \
                                f"답변은 'YYYYMMDD,YYYYMMDD' 형식으로 연월일을 의미하는 8자리 숫자를 ','(반점)으로 끊어서 표현해야 합니다. \n\n" \
                                f"주어진 문장 : {str(date_info)}\n 이제 시작일,끝일 답을 구하시오:"
                inputs = tokenizer(query_ext_dates, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_length=100)
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 정규 표현식으로 날짜 추출
                date_pattern = re.compile(r'(\d{8})')
                matches = date_pattern.findall(decoded_output)
                if len(matches) >= 2:
                    start_date, end_date = matches[0], matches[1]
                else:
                    # 날짜 추출 실패 시 기본값 사용
                    start_date, end_date = "20181231", "20181231"
                
        print("Denote start_date:::", start_date)
        print("Denote end_date:::", end_date)
        # 날짜 범위에 해당하는 인덱스 병합
        merged_db = self.merge_date_range(
            start_date,
            end_date,
            use_metadata=use_metadata,
            output_name=f"merged_{start_date}_to_{end_date}" if use_metadata else f"merged_{start_date}_to_{end_date}_nm"
        ) #      '''.strftime("%Y%m%d")'''

        if merged_db:
            print("QUERY PRINT::: ", query)
            # 유사도 측정
            sample_sentence="오늘 화재가 발생한 장소?"
            q_embed = self.embeddings.embed_query(query)
            sample_embed = self.embeddings.embed_query(sample_sentence)
            # q_embed와 sample_embed의 유사도 측정
            # q_distance = faiss.pairwise_distances(q_embed, sample_embed)
            distance = np.linalg.norm(np.array(q_embed) - np.array(sample_embed))
            print("q_distance:::", distance)
            s = merged_db.similarity_search_with_score(query, k=k)
            # 유사도 점수를 기준으로 내림차순 정렬
            # 점수를 유사도로 변환
            s = [(doc, 1 / (1 + score)) for doc, score in s]

            # 유사도를 기준으로 내림차순 정렬 (클수록 유사도가 높음)
            # s_sorted = sorted(s, key=lambda x: x[1], reverse=True)
            return s
        return []
