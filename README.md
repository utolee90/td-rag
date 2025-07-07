##  RTRAG 개발 키트

### 설치 패키지
`> pip install -r requirements.txt`

### 서버 실행 방법
`> (CUDA_VISIBLE_DEVICES=8) python run.py (port번호)`

### 추가 작성 사항
* `keys.py`에서 작성
```
    GCS_KEY = (Google Custom Search 작성)
    ENGINE_KEY = (Google Search Engine 키)
    openai_api_key= (OPENAI API 키)
    MODEL_PATH = (MODEL 경로) # 모델 경로 설정
    MODEL_NAMES = [ (사용 모델명) ]
```
### 코드 작성 상황
```
    data - data corpus 작성
    faiss_indexes - FAISS 기반 document 저장
    generation - 생성 모델용 (현재 비어있음)
    logs - 로그파일 저장. (업데이트 안함)
    resutls - 결과 저장
    retrieval - 검색자(Retriever) 관련 코드 (DPR, GCS 등)
    evaluate.py - 점수 평가 관련
    keys.py (사용 키 관련)
    manager.py (Vector Store Manager 관련 코드)
    run.py (Gradio 실행 어플리케이션)
    search.py (검색 관련 어플리케이션)
    utils.py (데이터처리 보조 등)
```

### 사용방법
 * http://165.132.192.52:7860에 접속
 * 모델 타입 선택 (Local 또는 OpenAI)
   * 기본키는 별도로 내장되어 있으며, 사용자 OpenAI 키 입력 후 Initialize Model 버튼 눌러서 키 연결 가능
 * 특정한 질문에 대해 답변 확인 방법
   * 우선 Please Enter your query 부분에 답변하고 싶은 질문 입력
   * 그 다음에 number of results에서 Retriever가 가져올 답변 선택
   * 마지막으로 date information에서 날짜 범위 지정. (시작일)/(끝일) - YYYYMMDD 형식으로
   * Retrieve from query 버튼을 누르면 관련 질문에 대한 Retrieving 결과 가져옴
   * Search 버튼을 누르면 관련 질문에 대한 답변 검색
   * Retriever 타입 선택 가능 (추후 지원예정)
* Answer - 답변 출력
* Explanation of Search Resutls - 답변과 함께 근거가 되는 뉴스 검색결과 출력
* Check Index Status - 날짜별 FAISS 인덱스 정보 검색
* File Upload For Retrieval and Saving for FAISS
    * QA 파일을 가져와서 드래그 하면 각 QA에 맞게 질문을 추출한 뒤 Retriever를 통해 답변 검색 결과 Document를 FAISS DB에 저장
* QA Answer Retrieval - QA 파일을 가져와서 드래그 하면 QA에 대해 답변 수행.


### 개선 중인 사항
 * 다른 Retriever 사용이 가능하게 개선 (현재는 Google Custom Search + DPR만 사용가능)
 * 로컬 모델에서 잘 돌아가는지 검증 중

### 저작자 현황
utopiamath 