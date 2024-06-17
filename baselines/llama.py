from rouge_score import rouge_scorer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import math
import jsonlines

import torch
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def save_arr(arr, save_path):
    with jsonlines.open(save_path, "w") as writer:
        for d in arr:
            writer.write(d)

def read_josnl(data_path):
    contents = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            contents.append(obj)
    return contents

if __name__ == "__main__":

    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }
    dataset = "cnn_dailymail"
    dataset_path = "data/{}/test_random1.jsonl".format(dataset)
    test_data = read_josnl(dataset_path)

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"  # meta-llama/Meta-Llama-3-8B-Instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = LLM(
        model=model_name_or_path,
        dtype=torch.bfloat16,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        max_tokens=512,
        skip_special_tokens=True
    )
    SUMMARY_SYSTEM_PROMPT = """Generate a concise and coherent summary towards the given article and don't generate anything else. Make sure the summary is clear, informative, and well-structured.\n\n"""

    def apply_chat_template_to_raw_prompts(prompt,system_prompt=SUMMARY_SYSTEM_PROMPT):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  prompt}
        ], tokenize=False)
    dataset_specific_prompt = "Summarize the article in three sentences around 60 words."
    if dataset == "xsum": dataset_specific_prompt = "Summarize the article in one sentences around 35 words."

    def post_process_llm_data(article):
        words = article.split(" ")
        article = " ".join(words[:2048])
        user_prompt = "Article: {article}. \n\n {dataset_specific_prompt} \n\n Summary:".format(article=article, dataset_specific_prompt=dataset_specific_prompt)
        outputs = llm.generate(apply_chat_template_to_raw_prompts(user_prompt), sampling_params, use_tqdm=False)
        llm_genrated_resp = outputs[0].outputs[0].text
        return llm_genrated_resp.split("\n")[-1].strip()
    
    test_summary_result = []
    for sample in test_data:
        article = sample[summarization_name_mapping[dataset][0]]
        summary = post_process_llm_data(article)
        test_summary_result.append({
            "pred": summary,
            "refer": sample["pseudo_summary"]
        })    
    save_arr(test_summary_result, "logs/results/llama/{}.jsonl".format(dataset))
