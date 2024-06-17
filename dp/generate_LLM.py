import os
import sys

sys.path.append(os.getcwd())

from rouge_score import rouge_scorer
from datasets import load_from_disk
from tqdm import tqdm
from utils import save_arr
import math

import torch

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--split", action='store_true')
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

    if args.split:
        ration = 0.8
        total_len = len(train_data)
        gsg_len = math.ceil(total_len*ration)
        llm_raw_data = train_data.select(range(gsg_len, total_len))
    else:
        llm_raw_data = train_data

    pseudo_summary_generated_by_llm = []
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = LLM(
        model=model_name_or_path,
        dtype=torch.bfloat16,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
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
    if dataset == "xsum": dataset_specific_prompt = "Summarize the article in one sentence around 35 words."
    if dataset == "xlsum": dataset_specific_prompt = "Summarize the article in two sentences around 22 words."
    def post_process_llm_data(article):
        words = article.split(" ")
        article = " ".join(words[:2048])
        user_prompt = "Article: {article}. \n\n {dataset_specific_prompt} \n\n Summary:".format(article=article, dataset_specific_prompt=dataset_specific_prompt)
        outputs = llm.generate(apply_chat_template_to_raw_prompts(user_prompt), sampling_params, use_tqdm=False)
        llm_genrated_resp = outputs[0].outputs[0].text
        return llm_genrated_resp.split("\n")[-1].strip()
    
    # 全部生成 便于之后做ablation（全部都用llm生成的 和 extract+abstract参杂 那个更好）
    save_path = "data/{}/{}_total.jsonl".format(dataset, "llm")
    if args.split:
        save_path = "data/{}/{}_split.jsonl".format(dataset, "llm")
    idx = 0
    for data_sample in tqdm(llm_raw_data):
        article = data_sample[summarization_name_mapping[dataset][0]]
        pseudo_summary = post_process_llm_data(article)
        data_sample["pseudo_summary"] = pseudo_summary
        pseudo_summary_generated_by_llm.append(data_sample)
        idx += 1
        if idx % 3000 == 0:
            print("save {} sample".format(idx))
            save_arr(pseudo_summary_generated_by_llm, save_path)
    save_arr(pseudo_summary_generated_by_llm, save_path)
    