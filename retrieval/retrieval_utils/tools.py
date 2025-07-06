import json, jsonlines, datetime, string, re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
        
def read_jsonl(in_file):
    questions = []
    with open(in_file, encoding = 'utf8') as fin:
        for line in fin:
            question = json.loads(line)
            questions.append(question)
    return questions

def fall_back(data_1, data_2, top_k=5):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]
        datum_1["search_result"] = [article for article in datum_1["search_result"] if "text" in article]
        nb_retrieved = len(datum_1["search_result"])
        if nb_retrieved < top_k:
            datum_1["search_result"].extend(datum_2["search_result"][:top_k-nb_retrieved])
    return data_1

def answer2jsonl(answers, questions, out_file, scores = None):
    # Confirm we have answers for all questions
    assert len(answers) == len(questions)
    if scores is not None:
        assert len(answers) == len(scores)
    outputs = []
    for q_idx in range(len(answers)):
        if scores is None:
            output = {"question_id": questions[q_idx]["question_id"], "prediction" : answers[q_idx]}
        else:
            output = {"question_id": questions[q_idx]["question_id"], "prediction" : answers[q_idx], "score" : scores[q_idx]}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)

def wiki2jsonl(questions, retrieved_docs, out_file):
    outputs = []
    assert len(questions) == len(retrieved_docs)
    for question, retrieved in zip(questions, retrieved_docs):
        search_result = []
        for doc_id, doc, doc_scores, publish_date in zip(retrieved["doc_ids"], retrieved["docs"], retrieved["doc_scores"], retrieved["publish_dates"]):
            search_result.append({"doc_id" : doc_id, "text" : doc, "doc_scores": doc_scores, "publish_date" : publish_date})
        output = {"question_id" : question["question_id"], "search_result" : search_result}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)

def check_jsonls(data_1, data_2):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]

def add_today(sentence, date):
    date = datetime.datetime.strptime(date, '%Y/%m/%d')
    date = date.strftime("%B %d, %Y")
    sentence = "Today is {}. ".format(date) + sentence
    return sentence

def normalize_answer(s):
    # TODO: should we keep those counter removal? 
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# 답변 판단 기준 함수
def get_answers_from_choices(response, choices, default=3):
    """
        응답 response
        답변 choices 
    """
    
    zero_resp = ["0", "zero", "first"]
    one_resp = ["1", "one", "second"]
    two_resp = ["2", "two", "thrid"]
    three_resp = ["3", "three", "fourth"]

    resp_zip = [zero_resp, one_resp, two_resp, three_resp]

    result = str(default) # 기본
    response = clean_text(response)
    choices = [clean_text(choice) for choice in choices]

    for idx, choice in enumerate(choices):
        set_result = False
        # resp_text 안에 있으면 강제 탈출
        for txt in resp_zip[idx]:
            if txt in response:
                set_result = True
                result = str(idx)
                break
        if set_result: break
    if not set_result:
        cos_sim_series, result = cos_vector(response, choices)
    
    # 0.5 초과할 때에만 
    if set_result or max(cos_sim_series) > 0.5:
        return str(result)
    else:
        return str(default)

        
    
# 문자열을 소문자로 변환 후 정규 표현식을 사용하여 알파벳과 숫자를 제외한 모든 문자 제거
def clean_text(text):
    
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    return text

# 키워드의 코사인 유사도 측정.
def cos_vector(keyword:str, choices:list):
    # TF-IDF 벡터라이저 초기화
    vectorizer = TfidfVectorizer()
    
    size = len(choices)
    choices.append(keyword)
    tfidf_matrix = vectorizer.fit_transform(choices) # 단어 추가

    # 코사인 유사도 계산
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cos_sim_series = [cos_sim[j, size] for j in range(size)]

    # 최대 인덱스 기준으로 찾기
    result = cos_sim_series.index(max(cos_sim_series))

    return cos_sim_series, result
