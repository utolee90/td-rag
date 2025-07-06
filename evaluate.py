
import itertools, string
from collections import Counter


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

def metric_max_over_ground_truths(metric, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

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
        em_total += metric_max_over_ground_truths(exact_match_score, prediction, golds)
        f1_total += metric_max_over_ground_truths(f1_score, prediction, golds)
        #if score_exists:
        #    score += float(pred["score"])
    return {'em': em_total/count, 'f1': f1_total/count}
            