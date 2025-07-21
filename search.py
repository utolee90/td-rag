from typing import List, Tuple
from manager import VectorStoreManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import logging
import openai
import os
import re
from pathlib import Path
from utils import create_chunks
from keys import MODEL_PATH, MODEL_NAMES

class SearchInterface:
    def __init__(self, vector_store_manager: VectorStoreManager):
        try:
            from run_metadata import global_generator_model  # 현재 파일에서 import
        except ImportError:
            try:
                from run import global_generator_model  # 백업으로 run.py에서 import
            except ImportError:
                global_generator_model = "OpenAI MCQ"  # 기본값 설정

        torch.cuda.empty_cache()
        self.search_manager = vector_store_manager
        self.history: List[Tuple[str, str]] = []
        
        # global_generator_model 안전 처리
        if isinstance(global_generator_model, list):
            self.model = global_generator_model[0] if global_generator_model else "OpenAI MCQ"
        elif not isinstance(global_generator_model, str):
            self.model = str(global_generator_model)
        else:
            self.model = global_generator_model  # 생성기용 모델 사용
        self.tokenizer = None
        self.model_type = None
        self.openai_api_key = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_P = MODEL_PATH + MODEL_NAMES[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_P , use_fast=False, trust_remote_code=True)
        
        # pad_token이 없는 경우 eos_token을 pad_token으로 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model은 이미 global_generator_model로 설정됨
        self.model_name = model_P

        # self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to('cuda:0')
        

    def cleanup_current_model(self):
        """현재 로드된 모델 정리"""
        try:
            if self.model is not None:
                # CUDA 메모리 정리
                if hasattr(self.model, "cuda"):
                    self.model.cpu()
                del self.model
                torch.cuda.empty_cache()

            if self.tokenizer is not None:
                del self.tokenizer

            self.model = None
            self.tokenizer = None
            logging.info("이전 모델 정리 완료")

            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            logging.error(f"모델 정리 중 오류 발생: {str(e)}")
            return False
    def init_openai_model(self, api_key: str) -> str:
        """OpenAI 모델 초기화"""
        try:
            # 이미 API 키가 설정되어 있는 경우
            #if self.openai_api_key == api_key and self.model_type == "openai":
            #    return "이미 OpenAI 모델이 초기화되어 있습니다."

            # 이전 모델 정리
            if 0:
                self.cleanup_current_model()

            logging.info("OpenAI 모델 초기화 시작")
            self.model_type = "openai"
            self.current_model_name = "gpt-3.5-turbo"  # 기본 모델
            self.openai_api_key = api_key

            # API 키 검증
            if not api_key or len(api_key.strip()) < 10:
                raise ValueError("Invalid OpenAI API key provided. Please check your API key.")

            # 테스트 요청으로 API 키 검증
            try:
                client = openai.OpenAI(api_key=api_key)
                client.chat.completions.create(
                    model=self.current_model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
            except Exception as api_e:
                print("api_key: ", api_key)
                raise ConnectionError(f"API Connection Failed: {str(api_e)}")

            logging.info("Initialized OpenAI Model successfully")
            return "Initialized OpenAI Model successfully"

        except Exception as e:
            error_msg = f"Fail to initialize OpenAI Model: {str(e)}"
            logging.error(error_msg)
            # 에러 발생 시 cleanup
            self.cleanup_current_model()
            self.openai_api_key = None
            self.model_type = None
            return error_msg

    def init_local_model_generate(self, model_path: str, model_name: str) -> str:
        """로컬 Hugging Face 모델 초기화"""
        try:
            from retrieval.dpr import load_model  # 지연 로드로 순환 참조 방지

            self.cleanup_current_model()

            logging.info(f"새 모델 로드 시작: {model_name}")
            full_path = os.path.join(model_path, model_name)

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                full_path, use_fast=False, trust_remote_code=True
            )
            
            # pad_token이 없는 경우 eos_token을 pad_token으로 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # GPU 메모리 상태 확인
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(
                    0
                ).total_memory - torch.cuda.memory_allocated(0)
                logging.info(
                    f"사용 가능한 GPU 메모리: {available_memory / 1024**2:.2f}MB"
                )

            # 모델 로드 with 메모리 최적화 설정
            self.model = AutoModelForCausalLM.from_pretrained(
                full_path,
                device_map="auto",
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload",
            )

            # 모델 디바이스 이동
            self.model.to(self.device)
            self.current_model_name = model_name

            logging.info(f"모델 {model_name} 로드 완료")
            return f"모델 {model_name} 로드 완료"

        except Exception as e:
            error_msg = f"모델 로드 실패: {str(e)}"
            logging.error(error_msg)
            # 에러 발생 시 cleanup 실행
            self.cleanup_current_model()
            return error_msg

    def init_local_model_mcq(self, model_path: str, model_name: str) -> str:
        """로컬 Hugging Face 모델 초기화"""
        try:
            # 같은 모델을 다시 로드하려는 경우 건너뛰기
            if 0:#self.current_model_name == model_name and self.model is not None:
                return f"이미 {model_name} 모델이 로드되어 있습니다."

            # 이전 모델 정리
            self.cleanup_current_model()

            logging.info(f"새 모델 로드 시작: {model_name}")
            full_path = os.path.join(model_path, model_name)

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                full_path, use_fast=False, trust_remote_code=True
            )

            # GPU 메모리 상태 확인
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(
                    0
                ).total_memory - torch.cuda.memory_allocated(0)
                logging.info(
                    f"사용 가능한 GPU 메모리: {available_memory / 1024**2:.2f}MB"
                )

            # 모델 로드 with 메모리 최적화 설정
            self.model = AutoModelForSequenceClassification.from_pretrained(
                full_path,  num_labels=4,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload",
            )

            self.model.to(self.device)
            self.current_model_name = model_name

            logging.info(f"모델 {model_name} 로드 완료")
            return f"모델 {model_name} 로드 완료"

        except Exception as e:
            error_msg = f"모델 로드 실패: {str(e)}"
            logging.error(error_msg)
            # 에러 발생 시 cleanup 실행
            self.cleanup_current_model()
            return error_msg

    def generate_answer(self, model_type, question: str, context: str, choices=None) -> str:
        if model_type in ["Local HuggingFace - Generate",]:

            if 1:
                # 더 효과적인 프롬프트
                prompt = f"""Given the following information, answer the question directly and concisely.

Information:
{context if context else "No context provided."}

Question: {question}

Answer:"""
            
            # 디버깅: 프롬프트 내용 확인
            print(f"[DEBUG] Final prompt length: {len(prompt)}")
            print(f"[DEBUG] Final prompt preview:")
            print(f"--- PROMPT START ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print(f"--- PROMPT END ---")

            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True, padding=True)
            inputs = inputs.to(self.model.device)
            
            # 디버깅: 토크나이저 정보 확인
            print(f"[DEBUG] Input tokens length: {inputs['input_ids'].shape}")
            print(f"[DEBUG] Tokenizer pad_token: {self.tokenizer.pad_token}")
            print(f"[DEBUG] Tokenizer eos_token: {self.tokenizer.eos_token}")
            
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
        # Generation parameters - 균형 잡힌 설정
            generation_config = {
                "max_new_tokens": 50,   # 간단한 답변을 위해 줄임
                "temperature": 0.3,     # 적절한 수준의 창의성
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

            outputs = self.model.generate(
                **inputs,
                **generation_config
            )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 디버깅: 생성된 전체 답변 확인
            print(f"[DEBUG] Raw generated text length: {len(answer)}")
            print(f"[DEBUG] Raw generated text:")
            print(f"--- ANSWER START ---")
            print(answer)
            print(f"--- ANSWER END ---")
            
            # 프롬프트 부분 제거하여 실제 답변만 추출
            if prompt in answer:
                actual_answer = answer.replace(prompt, "").strip()
                print(f"[DEBUG] Extracted answer: {actual_answer}")
                return actual_answer
            
            return answer

        elif model_type == ["Local HuggingFace - MCQ",]:

            # 1. System prompt
            system_prompt = """You are an expert in solving multiple-choice questions based on given contexts.
    Rules:
    - Judge based only on the provided context
    - Analyze each choice systematically
    - Select the most appropriate single answer
    - If unsure, state the reason clearly"""

            # 2. Task prompt
            task_prompt = """Please follow these steps to answer the question:
    1) Understand the question: Identify what is being asked
    2) Analyze the context: Check relevant information
    3) Review choices: Evaluate the accuracy of each option
    4) Select answer: Choose the most appropriate answer"""

            # 3. Format choices
            formatted_choices = "\n".join([f"{key}. {value}" for key, value in choices.items()])

            # 4. Input formatting
            input_prompt = f"""Question: {question}

    Given choices:
    {formatted_choices}

    Given context:
    {context if context else "No context provided."}

    Answer for analyzing:"""

            # 5. Final prompt composition
            prompt = f"{system_prompt}\n\n{task_prompt}\n\n{input_prompt}"

            # Tokenizer setup
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            )
            inputs = inputs.to(self.model.device)

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            # Model prediction
            outputs = self.model(**inputs)

            # For sequence classification, get the predicted class
            predictions = outputs.logits.softmax(dim=-1)
            predicted_class = predictions.argmax().item()

            # Convert numeric prediction to choice key
            choice_keys = list(choices.keys())
            selected_answer = choice_keys[predicted_class]

            # Format final answer
            final_answer = f"""Selected answer: {selected_answer}
    Reason: The answer {choices[selected_answer]} is supported by the context provided."""

            return final_answer

        elif model_type == "OpenAI":
            if self.openai_api_key is None:
                return "No OpenAI API key provided. Please set the API key first."

            return self._process_openai_generate(question, context)

        elif model_type == "OpenAI MCQ":
            if self.openai_api_key is None:
                return "No OpenAI API key provided. Please set the API key first."

            return self._process_openai_mcq(question, context, choices)

        else:
            return "Unsupported model type. Please select a valid model type."

    def search_news(self, model_type, query: str, top_k: int, date_info: str, pos=True, use_metadata=True) -> str:
        if not query.strip():
            return "Please enter a valid query."
        
        # model_type 안전 처리
        if isinstance(model_type, list):
            model_type = model_type[0] if model_type else "OpenAI MCQ"
        elif not isinstance(model_type, str):
            model_type = str(model_type)
        
        try:
            if date_info is None or date_info.strip() in ["", "None", "20000000/20000000"]:
                results = []
            else:
                results = self.search_manager.search_with_date(query, k=top_k, date_info=date_info, use_metadata=use_metadata)
                print("results", results)
            
            output = []
            # 결과가 없을 경우
            if not results and pos:
                return "No news articles found for the given query and date range."
            

            # 관련 문서들의 내용을 하나로 합치기
            all_content = "\n\n".join([doc.page_content for doc, _ in results])
            
            # LLM을 사용한 키워드 추출 함수
            def extract_keywords_with_llm(question):
                """LLM을 사용해서 질문에서 중요한 키워드 추출"""
                try:
                    keyword_prompt = f"""Given the following question, extract 3-5 important keywords or key phrases that would be most relevant for finding the answer in a knowledge base.
Focus on entities, concepts, and descriptive terms that are essential to answer the question.
Return only the keywords separated by commas, nothing else.

Question: {question}
Keywords:"""
                    
                    if model_type in ["Local HuggingFace - Generate"]:
                        # Local LLM 사용
                        inputs = self.tokenizer(keyword_prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=50,
                                temperature=0.1,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        
                        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        keywords_response = full_response.replace(keyword_prompt, "").strip()
                        keywords = [k.strip().lower() for k in keywords_response.split(',') if k.strip()]
                        print(f"[DEBUG] Extracted keywords from LLM: {keywords}")
                        return keywords[:5]  # 최대 5개
                        
                    elif model_type in ["OpenAI", "OpenAI MCQ"]:
                        # OpenAI API 사용
                        if self.openai_api_key is None:
                            print("[DEBUG] No OpenAI API key available for keyword extraction")
                            # 폴백 사용
                            simple_keywords = [word.lower() for word in question.split() if len(word) > 2]
                            print(f"[DEBUG] Using fallback keywords: {simple_keywords}")
                            return simple_keywords
                            
                        client = openai.OpenAI(api_key=self.openai_api_key)
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": keyword_prompt}],
                            max_tokens=50,
                            temperature=0.1
                        )
                        keywords_response = response.choices[0].message.content.strip()
                        keywords = [k.strip().lower() for k in keywords_response.split(',') if k.strip()]
                        print(f"[DEBUG] Extracted keywords from OpenAI: {keywords}")
                        return keywords[:5]  # 최대 5개
                        
                except Exception as e:
                    print(f"[DEBUG] Error extracting keywords with LLM: {e}")
                    # 폴백: 간단한 키워드 추출
                    simple_keywords = [word.lower() for word in question.split() if len(word) > 2]
                    print(f"[DEBUG] Using fallback keywords: {simple_keywords}")
                    return simple_keywords
            
            
            def extract_relevant_context(content, question, max_length=3000):
                """질문과 관련된 핵심 내용을 추출"""
                question_lower = question.lower()
                question_keywords = question_lower.split()

                # LLM을 사용해서 중요한 키워드 동적 추출
                important_keywords = extract_keywords_with_llm(question)

                # 모든 문서에서 콘텐츠만 추출
                all_contents = []
                for doc_content in content.split('\n\n'):
                    if len(doc_content.strip()) < 50:  # 너무 짧은 내용 제외
                        continue

                    # Title과 Contents 분리
                    if 'Title:' in doc_content and 'Contents:' in doc_content:
                        parts = doc_content.split('Contents:', 1)
                        if len(parts) == 2:
                            title = parts[0].replace('Title:', '').strip()
                            content_text = parts[1].strip()

                            # 제목과 내용을 결합
                            all_contents.append(f"Title: {title}\nContent: {content_text}")
                        else:
                            all_contents.append(doc_content)
                    else:
                        # Title/Contents 구분이 없는 경우 전체를 추가
                        all_contents.append(doc_content)

                # 콘텐츠를 하나로 결합
                final_context = '\n\n'.join(all_contents)
                print(f"[DEBUG] Final context length: {len(final_context)}")
                return final_context

            # 스마트 컨텍스트 추출
            if all_content.strip() == "":
                context = ""
            else:
                context = extract_relevant_context(all_content, query)
            
            # 디버깅: 컨텍스트 내용 확인
            print(f"[DEBUG] Original content length: {len(all_content)}")
            print(f"[DEBUG] Extracted context length: {len(context)}")
            print(f"[DEBUG] Number of documents in context: {len(results)}")
            print(f"[DEBUG] Extracted context:")
            print(f"--- EXTRACTED CONTEXT START ---")
            print(context)
            print(f"--- EXTRACTED CONTEXT END ---")


            # context_chunks = create_chunks(context, chunk_size=1000, overlap=500)

            # 결과만 출력
            # answer_list = []
            if not pos:

                print(f"[DEBUG] Find Model Type and Query: {model_type}, {query}")
                answer = self.generate_answer(model_type, query, context)
                print(f"[DEBUG] Generated answer: {answer[:100]}...")  # 답변 미리보기
                output.append(f"{answer}")
                return "\n".join(output)
            
            # 프로세스 전체 출력
            output.append(f"Question: {query}\n")
            output.append(f"Number of searched documents: {len(results)}\n")
            output.append("-" * 80 + "\n")
            output.append("\n" + "-" * 80 + "\n")

            answer = self.generate_answer(model_type, query, context)

            output.append("?? Questions and answers:")
            output.append("\n" + "-" * 80 + "\n")

            output.append(answer)
            output.append("\n" + "-" * 80 + "\n")

            output.append("?? References:")
            for i, (doc, score) in enumerate(results, 1):

                print("doc_page_content")
                print(doc.page_content)
                print()
                title = self.get_contents(doc.page_content, text_type='title')
                contents = self.get_contents(doc.page_content, text_type='contents')

                output.append(f"[{i}] Similarity score: {score:.4f}")
                output.append(f"Date: {doc.metadata['date']}")
                output.append(f"Source: {doc.metadata['source']}")
                output.append(f"Title: {title}")
                output.append(f"URL: {doc.metadata['url']}")
                output.append(f"Contents: {contents}")
                output.append("-" * 80 + "\n")

            return "\n".join(output)

        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return f"An error occurred during searching: {str(e)}"

    @staticmethod
    def get_contents(page_content, text_type='contents'):
        """
        Extracts the contents from a document based on the specified type.
        :param doc: The document object containing metadata and content.
        :param type: The type of content to extract ('contents', 'title', 'url').
        :return: The extracted content as a string.
        """
        # 디버깅: page_content 형태 확인
        print(f"[DEBUG] get_contents called with text_type: {text_type}")
        print(f"[DEBUG] page_content full content:")
        print(f"--- PAGE CONTENT START ---")
        print(page_content)
        print(f"--- PAGE CONTENT END ---")
        
        # 일단 전체 content를 반환하도록 단순화
        if text_type == 'title':
            # 첫 번째 줄이나 짧은 제목 추출 시도
            lines = page_content.split('\n')
            first_line = lines[0] if lines else page_content
            # 너무 길면 앞부분만 사용
            return first_line[:100] + "..." if len(first_line) > 100 else first_line
        elif text_type == 'contents':
            # 전체 내용 반환
            return page_content
        else:
            return f"Invalid text_type: {text_type}. Use 'title' or 'contents'."
    
    def _process_openai_generate(self, question: str, context: str) -> str:
        """OpenAI Generate 모드: 자유 형태 답변 생성"""
        try:
            max_context = context

            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Please provide a detailed and comprehensive answer based on the provided context. If the context doesn't contain enough information, clearly state what information is missing and provide the best answer you can with available information."
                },
                {
                    "role": "user", 
                    "content": f"Context: {max_context if max_context else 'No specific context provided'}\n\nQuestion: {question}\n\nPlease provide a detailed answer:"
                }
            ]

            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 더 나은 성능을 위해 gpt-4 사용
                messages=messages,
                temperature=0.7,
                max_tokens=800,  # Generate 모드는 더 긴 답변 허용
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

    def _process_openai_mcq(self, question: str, context: str, choices:str) -> str:
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
                system_prompt = "You are an expert assistant. Based on the provided context, give a concise and precise answer. If it's a factual question, provide a specific answer. Keep responses focused and to the point."
                user_prompt = f"Context: {max_context if max_context else 'No specific context provided'}\n\nQuestion: {question}\n\nProvide a concise, precise answer:"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,  # MCQ는 더 일관된 답변을 위해 낮은 temperature
                max_tokens=300,   # MCQ는 짧은 답변
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