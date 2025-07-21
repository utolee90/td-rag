from utils import analyze_qa_type

qs_obj = {"question_id": "20220610_0_nota", "question_date": "2022/06/10", "question_source": "CNN", "question_url": "https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/", "question_sentence": "According to a recent ranking, which is the world’s most expensive city?", "choices": ["New York", "Tokyo", "London", "None of the above"], "answer": ["3"], "evidence": "For the third year in a row, <a href=\"https://www.cnn.com/travel/article/world-most-expensive-cities-2022-intl-hnk/index.html\">Hong Kong</a> received the dubious title of “priciest city in the world,” according to an analysis this week by global mobility company ECA International. The company calculates the list based on several factors, including the average price of groceries, rent, utilities, public transit and the strength of the local currency."}
qa_name = "realtimeqa"
question_type = "MCQ"
use_type = "Retriever Only"

# print(analyze_qa_type(qs_obj, qa_name, question_type, use_type))

import json
qs_result = analyze_qa_type(qs_obj, qa_name, question_type, use_type)
json_str = json.dumps(qs_result, ensure_ascii=False, indent=4)
# print(json_str)

answer_obj = [{'id': '20220610_0_nota', 'query': 'According to a recent ranking, which is the world’s most expensive city?', 'score': 0.5, 'answer': ['No specific answer available.'], 'prediction': ['No specific answer available.']}, {'id': '20220610_1_nota', 'query': 'The baby formula manufacturing plant at the center of the nationwide shortage recently restarted production of certain formulas. Why was the facility closed for several months?', 'score': 0.5, 'answer': ['The facility was closed due to safety concerns and contamination issues.'], 'prediction': ['The facility was closed due to safety concerns and contamination issues.']}, {'id': '20220610_2_nota', 'query': 'Which golfer won the 2022 US Women’s Open?', 'score': 0.5, 'answer': ['Minjee Lee'], 'prediction': ['Minjee Lee']}, {'id': '20220610_3_nota', 'query': 'According to a recent California ruling, which insect can legally be considered a fish and have the same protections?', 'score': 0.5, 'answer': ['The California ruling allows the California red-legged frog to be considered a fish.'], 'prediction': ['The California ruling allows the California red-legged frog to be considered a fish.']}, {'id': '20220610_4_nota', 'query': 'Which department store chain put itself up for sale this week?', 'score': 0.5, 'answer': ['No specific answer available.'], 'prediction': ['No specific answer available.']}]
news_data = [{'question_id': '20220610_0_nota', 'question_date': '2022/06/10', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/', 'question_sentence': 'According to a recent ranking, which is the world’s most expensive city?', 'choices': ['New York', 'Tokyo', 'London', 'None of the above'], 'answer': ['3'], 'evidence': 'For the third year in a row, <a href="https://www.cnn.com/travel/article/world-most-expensive-cities-2022-intl-hnk/index.html">Hong Kong</a> received the dubious title of “priciest city in the world,” according to an analysis this week by global mobility company ECA International. The company calculates the list based on several factors, including the average price of groceries, rent, utilities, public transit and the strength of the local currency.'}, {'question_id': '20220610_1_nota', 'question_date': '2022/06/10', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/', 'question_sentence': 'The baby formula manufacturing plant at the center of the nationwide shortage recently restarted production of certain formulas. Why was the facility closed for several months?', 'choices': ['Employee walkouts', 'Bacteria outbreak', 'Supply chain issues', 'None of the above'], 'answer': ['1'], 'evidence': 'Abbott’s Michigan-based facility was shut down for months following an FDA inspection that found <a href="https://www.cnn.com/2022/06/04/health/abbott-formula-plant-restarts/index.html">dangerous bacteria</a> – which can be deadly to infants – in several areas of the plant.'}, {'question_id': '20220610_2_nota', 'question_date': '2022/06/10', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/', 'question_sentence': 'Which golfer won the 2022 US Women’s Open?', 'choices': ['Minjee Lee', 'Mina Harigae', 'Jin Young Ko', 'None of the above'], 'answer': ['0'], 'evidence': '<a href="https://www.cnn.com/2022/06/05/golf/minjee-lee-us-womens-open/index.html">Minjee Lee</a> won the US Women’s Open. It’s the second major win for the 26-year-old golf star.'}, {'question_id': '20220610_3_nota', 'question_date': '2022/06/10', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/', 'question_sentence': 'According to a recent California ruling, which insect can legally be considered a fish and have the same protections?', 'choices': ['Spider', 'Ladybug', 'Bumblebee', 'None of the above'], 'answer': ['2'], 'evidence': 'A California court has ruled <a href="https://www.cnn.com/2022/06/06/us/california-bees-fish-court-ruling-scn-trnd/index.html#:~:text=(CNN)%20A%20fishy%20ruling%20from,the%20California%20Endangered%20Species%20Act">bees can legally be considered fish</a> under specific circumstances. The expansion of the definition of fish to include invertebrates makes bees eligible for greater protection from the Fish and Game Commission, wrote the court.'}, {'question_id': '20220610_4_nota', 'question_date': '2022/06/10', 'question_source': 'CNN', 'question_url': 'https://edition.cnn.com/interactive/2022/06/us/cnn-5-things-news-quiz-june-9-sec/', 'question_sentence': 'Which department store chain put itself up for sale this week?', 'choices': ['Nordstrom', 'Kohl’s', 'JCPenney', 'None of the above'], 'answer': ['1'], 'evidence': 'Kohl’s announced Monday it has entered into a negotiation period for a <a href="https://www.cnn.com/2022/06/07/business/kohls-franchise-group-offer/index.html">potential sale</a>. Franchise Group, a holding company that owns a number of retail brands, has proposed to buy Kohl’s in a deal valued at around $8 billion.'}]

from evaluate import *

print(gen_eval(answer_obj, news_data))

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
                res_obj["em"] = exact_match_score(pred[0], golds[0])
                res_obj["f1"] = f1_score(pred[0], golds[0])
        accuracy_results.append(res_obj)
    
    accuracy_objs = [json.dumps(res, ensure_ascii=False) for res in accuracy_results]

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write("\n".join(accuracy_objs))
    
    print(f"Accuracy report saved to {file_name}")
    return accuracy_results

accuracy_report_file = "sample_accuracy_report.jsonl"
accuracy_reports = make_accuracy_reports(answer_obj, news_data, file_name=accuracy_report_file)