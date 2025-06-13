# Валидация через BERT score

from bert_score import score
def evaluate(predictions, references):
    P, R, F1 = score(predictions, references, lang="ru")
    return {"bert_precision": P.mean(), "bert_recall": R.mean(), "bert_f1": F1.mean()}