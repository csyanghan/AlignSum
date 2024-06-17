import os
import sys

sys.path.append(os.getcwd())

from utils import save_arr
import random
import json
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cnn_dailymail")
    args = parser.parse_args()

    dataset_name = args.dataset

    specific_datapath = "data/{}/{}_element_aware.json".format(dataset_name, dataset_name)

    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }

    with open(specific_datapath, "r") as fp:
        specific_data = json.load(fp)[dataset_name]

    random.shuffle(specific_data)


    specific_data_train = list(map(lambda x: ({summarization_name_mapping[dataset_name][0]: x["src"], "id": str(x["id"]+1), summarization_name_mapping[dataset_name][1]: x["original_summary"], "pseudo_summary": x["element-aware_summary"]}), specific_data[:100])) 
    specific_data_test = list(map(lambda x: ({summarization_name_mapping[dataset_name][0]: x["src"], "id": str(x["id"]+1), summarization_name_mapping[dataset_name][1]: x["original_summary"], "pseudo_summary": x["element-aware_summary"]}), specific_data[100:]))
    save_arr(specific_data_train, "data/{}/train_{}.jsonl".format(dataset_name, "random1"))
    save_arr(specific_data_test, "data/{}/test_{}.jsonl".format(dataset_name, "random1"))
