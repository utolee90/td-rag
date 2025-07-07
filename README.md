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

### 저작자 현황
utopiamath 