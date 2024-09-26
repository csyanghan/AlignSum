import os
import sys
import argparse
sys.path.append(os.getcwd())

from tqdm import tqdm
from utils import read_jsonl, save_arr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn")
    args = parser.parse_args()
    dataset = args.dataset

    # https://huggingface.co/facebook/bart-large-xsum
    # https://huggingface.co/facebook/bart-large-cnn
    device = torch.device("cuda:1")
    summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-{}".format(dataset)).to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-{}".format(dataset))

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
        inputs = tokenizer([article], max_length=1024, return_tensors="pt", truncation=True)
        # Generate Summary
        summary_ids = summarizer.generate(inputs["input_ids"].to(device))
        pred = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        sample["bart_ft"] = pred

    save_arr(specific_data, "./logs/results/bart_ft/{}.jsonl".format(dataset))
