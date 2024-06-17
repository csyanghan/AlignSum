import os
import sys

sys.path.append(os.getcwd())

from rouge_score import rouge_scorer
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from utils import save_arr
import math

from multiprocessing import Process, Queue
import argparse

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def principal_GSG(article, topk=3):
    sentences = article.split(".")
    # 每个句子大于6个单词才认为是一个有效句子
    clean_sentences = [ sen + "." for sen in sentences if len(sen.split(" ")) > 6]
    principal_sens = []
    for _ in range(topk):
        scores_f1 = []
        for cur_sentence in clean_sentences:
            left_sentences = []
            for sen in sentences:
                if sen not in principal_sens and sen != cur_sentence:
                    left_sentences.append(sen)
            left_article = "".join(left_sentences)
            selected_sentences = "".join(principal_sens + [cur_sentence])
            cur_scores = scorer.score(selected_sentences, left_article)
            scores_f1.append(cur_scores["rouge1"][2])
        if len(scores_f1) == 0: continue
        max_idx = np.argmax(scores_f1)
        principal_sens.append(clean_sentences[max_idx])
        clean_sentences.pop(max_idx)
    return principal_sens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dailymail")
    args = parser.parse_args()

    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
        "xlsum": ("text", "summary"),
    }
    dataset = args.dataset
    dataset_path = "data/original/{}".format(dataset)
    full_data = load_from_disk(dataset_path)
    train_data = full_data["train"]

    ration = 0.8
    total_len = len(train_data)
    gsg_len = math.ceil(total_len*ration)
    gsg_raw_data = train_data.select(range(gsg_len))

    n_process = 16
    process_len = math.floor(gsg_len/n_process)
    process_list = []
    pseudo_summary_generated_by_principal_GSG = []
    q = Queue(maxsize=100)

    def process_func(q, start_idx, end_idx):
        print("Process start, process data from {} to {}".format(start_idx, end_idx))
        cur_process_data = gsg_raw_data.select(range(start_idx, end_idx))
        cur_process_pseudo_summary = []
        for data_sample in tqdm(cur_process_data):
            article = data_sample[summarization_name_mapping[dataset][0]]
            pseudo_summary = "".join(principal_GSG(article, topk=1))
            data_sample["pseudo_summary"] = pseudo_summary
            cur_process_pseudo_summary.append(data_sample)
        q.put(cur_process_pseudo_summary)

    for i in range(n_process):
        p = Process(target=process_func, args=(q, i*process_len, (i+1)*process_len, ))
        p.start()
        process_list.append(p)

    # 先把数据取走 不然数据变量太大会造成死锁
    for i in range(n_process):
        pseudo_summary_generated_by_principal_GSG += q.get()
    
    for p in process_list:
        p.join()
    
    print("Save to data/{}/{}.jsonl, length: {}".format(dataset, "gsg", len(pseudo_summary_generated_by_principal_GSG)))
    save_arr(pseudo_summary_generated_by_principal_GSG, "data/{}/{}.jsonl".format(dataset, "gsg"))
