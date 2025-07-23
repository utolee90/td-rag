
import itertools, string
from collections import Counter


def normalize_answer(s):
    # s가 리스트인 경우 첫 번째 요소 사용하거나 조인
    if isinstance(s, list):
        s = s[0] if s else ""
    elif not isinstance(s, str):
        s = str(s)
    
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

def metric_max_over_ground_truths(metric, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):

    match = []
    if isinstance(prediction, list):
        for pred, gold in zip(prediction, ground_truth):
            if normalize_answer(pred) == normalize_answer(gold):
                match.append(True)
            else:
                match.append(False)
        # match 중 true의 개수 비율 반환
        return sum(match) / len(match) if match else 0

    elif isinstance(prediction, str) and isinstance(ground_truth, str):
        return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    if isinstance(prediction, list):
        pred_match = []
        # 개별 예측에 대한 f1 점수 계산 후 평균
        for pred, gold in zip(prediction, ground_truth):
            if isinstance(pred, str) and isinstance(gold, str):
                pred_match.append(f1_score(pred, gold))
            else:
                pred_match.append(0)
        return sum(pred_match) / len(pred_match) if pred_match else 0
    
    if isinstance(prediction, str) and isinstance(ground_truth, str):
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

# 정확도 측정 # pred json list, gold json list
def accuracy(preds, golds):
    count = 0
    correct = 0
    score_exists = False
    if "score" in preds[0].keys():
        score_exists = True
        wrong_score = 0
        correct_score = 0
    for pred, gold in zip(preds, golds):
        prediction = pred["prediction"]
        gold = gold["answer"]
        if prediction == gold:
            correct += 1
            if score_exists:
                correct_score += float(pred["score"])
        else:
            if score_exists:
                wrong_score += float(pred["score"])
        count += 1
    if score_exists:
        wrong_score_val = wrong_score/(count-correct) if count-correct != 0 else 0
        correct_score_val = correct_score/correct if correct != 0 else 0
        return {'accuracy': correct/count, 'correct_score': correct_score_val, 'wrong_score': wrong_score_val, 'score': (correct_score + wrong_score)/count}
    return {'accuracy': correct/count}

def gen_eval(preds, golds):
    em_total = 0
    f1_total = 0
    count = 0
    #score = 0
    ##prediction = pred["prediction"].lower()
    #score_exists = False
    #if "score" in preds[0].keys():
    #    score_exists = True
    for pred, gold in zip(preds, golds):
        # Concatenate gold answers (can be multiple like NYT)
        sent = gold["question_sentence"].lower().strip()
        if "except" in sent[-10:]:
            continue
        count += 1
        golds = [gold["choices"][int(idx)] for idx in gold["answer"]]
        golds = [' '.join(perm) for perm in list(itertools.permutations(golds))]
        prediction = pred["prediction"]
        
        # prediction이 리스트인 경우 첫 번째 요소 사용하거나 조인
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else ""
        
        # golds = golds[0]
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, golds)
        f1_total += metric_max_over_ground_truths(f1_score, prediction, golds)
        #if score_exists:
        #    score += float(pred["score"])
    return {'em': em_total/count, 'f1': f1_total/count}
            