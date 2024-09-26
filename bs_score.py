from bert_score import score
from rouge import Rouge
from utils import read_jsonl
import evaluate

metric = evaluate.load("rouge")

def rouge_score(preds, refs):
    rouge = Rouge()
    rs = rouge.get_scores(preds, refs, avg=True)
    rouge1 = rs["rouge-1"]["f"] * 100
    rouge2 = rs["rouge-2"]["f"] * 100
    rougel = rs["rouge-l"]["f"] * 100
    return rouge1, rouge2, rougel

def evaluate_rouge_score(preds, refs):
    result = metric.compute(predictions=preds, references=refs, use_stemmer=True)    
    return result

def bs_score(preds, refs):
    # 保持和SumCoT一致 bert_score 默认使用 roberta-large 且层数为17
    _, _, F1 = score(preds, refs, model_type="FacebookAI/roberta-large", num_layers=17, device="cuda:1", lang="en", verbose=True)
    bs = F1.mean()
    return bs

if __name__ == "__main__":
    results = read_jsonl("path/to/results.jsonl")
    preds = list(map(lambda x: x["pred"], results))
    refs = list(map(lambda x: x["refer"], results))
    bs_score_f1 = bs_score(preds, refs)
    # rouge_score_metric = rouge_score(preds, refs)
    rouge_score_evaluate = evaluate_rouge_score(preds, refs)
    print(bs_score_f1, rouge_score_evaluate)
