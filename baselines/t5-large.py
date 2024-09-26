from transformers import pipeline

import os
import sys
import argparse
sys.path.append(os.getcwd())

from tqdm import tqdm
from utils import read_jsonl, save_arr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnndm")
    args = parser.parse_args()
    dataset = args.dataset

    device = torch.device("cuda:1")
    # https://huggingface.co/kssteven/T5-large-cnndm
    # https://huggingface.co/kssteven/T5-large-xsum
    tokenizer = AutoTokenizer.from_pretrained("kssteven/T5-large-{}".format(dataset))
    model = AutoModelForSeq2SeqLM.from_pretrained("kssteven/T5-large-{}".format(dataset)).to(device)

    dataset_path = "data/cnn_dailymail/test_random1.jsonl"
    if dataset == "xsum": dataset_path = "data/xsum/test_random1.jsonl"

    specific_data = read_jsonl(dataset_path)
    
    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }

    for sample in tqdm(specific_data):
        article_key = "article"
        if dataset == "xsum": article_key = "document" 
        article = sample[article_key]
        input_ids = tokenizer.encode(article, return_tensors='pt').to(device)
        summary_ids = model.generate(input_ids,
            min_length=20,
            max_length=140,
            do_sample=False
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        sample["t5_ft"] = summary_text

    save_arr(specific_data, "./logs/results/t5_ft/{}.jsonl".format(dataset))
