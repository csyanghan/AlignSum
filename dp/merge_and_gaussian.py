import os
import sys

sys.path.append(os.getcwd())

from utils import save_arr, read_jsonl
import random
from transformers import AutoTokenizer
import numpy as np
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dailymail")
    args = parser.parse_args()

    dataset_name = args.dataset


    gsg_path = "data/{}/gsg.jsonl".format(dataset_name)
    llm_path = "data/{}/llm.jsonl".format(dataset_name)
    human_annonated_path = "data/{}/train_random1.jsonl".format(dataset_name)

    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }
    article_key = summarization_name_mapping[dataset_name][0]

    tokenizer = AutoTokenizer.from_pretrained("bart-large")
    
    def get_sentence_tokens(sen):
        tokenizer_sen = tokenizer(sen)
        input_ids = tokenizer_sen["input_ids"]
        return len(input_ids)

    def get_target_distribution_mean_and_std():
        target_data = read_jsonl(human_annonated_path)
        target_data_token_distribution = [get_sentence_tokens(x["pseudo_summary"]) for x in target_data]
        target_data_mean = np.mean(target_data_token_distribution)
        target_data_std = np.std(target_data_token_distribution)
        return target_data_mean, target_data_std

    def gaussion_filter(data):
        mean, std_dev = get_target_distribution_mean_and_std()
        lower_bound = mean - 1.96 * std_dev
        upper_bound = mean + 1.96 * std_dev
        data_after_gaussian_filter = []
        for sample in data:
            pseudo_summary = sample["pseudo_summary"]
            pseudo_summary_len = get_sentence_tokens(pseudo_summary)
            if get_sentence_tokens(sample[article_key]) < 10:
                print("No document")
                continue
            if pseudo_summary_len > lower_bound and pseudo_summary_len < upper_bound:
                data_after_gaussian_filter.append(sample)
        return data_after_gaussian_filter


    def merge_dp():
        gsg_data = read_jsonl(gsg_path)
        llm_data = read_jsonl(llm_path)

        gsg_and_llm_data = gsg_data + llm_data
        random.shuffle(gsg_and_llm_data)
        save_arr(gsg_and_llm_data, "data/{}/gsg_llm.jsonl".format(dataset_name))

        human_annonated_data = read_jsonl(human_annonated_path)

        dp_data = gsg_and_llm_data + human_annonated_data
        random.shuffle(dp_data)
        save_arr(dp_data, "data/{}/dp_mix.jsonl".format(dataset_name))
        print("merge dp done!")


    def merge_dp_gaussian():
        gsg_data = read_jsonl(gsg_path)
        llm_data = read_jsonl(llm_path)

        gsg_data_filter = gaussion_filter(gsg_data)
        save_arr(gsg_data_filter, "data/{}/gsg_gaussian.jsonl".format(dataset_name))

        llm_data_filter = gaussion_filter(llm_data)
        save_arr(llm_data_filter, "data/{}/llm_gaussian.jsonl".format(dataset_name))
        
        gsg_and_llm_data = gsg_data + llm_data
        gsg_and_llm_data = gaussion_filter(gsg_and_llm_data)
        random.shuffle(gsg_and_llm_data)
        save_arr(gsg_and_llm_data, "data/{}/gsg_llm_gaussian.jsonl".format(dataset_name))

        human_annonated_data = read_jsonl(human_annonated_path)

        dp_data = gsg_and_llm_data + human_annonated_data
        random.shuffle(dp_data)
        save_arr(dp_data, "data/{}/dp_mix_gaussian.jsonl".format(dataset_name))
        print("merge dp guassian done!")

    def filter_llm_total_gaussian():
        llm_data_total = read_jsonl("data/xsum/llm_total.jsonl")
        llm_data_total = gaussion_filter(llm_data_total)
        save_arr(llm_data_total, "data/{}/llm_total_gaussian.jsonl".format(dataset_name))

    merge_dp()
    merge_dp_gaussian()
    # filter_llm_total_gaussian()
