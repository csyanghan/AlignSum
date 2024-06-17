import openai
import argparse
import tqdm
import time
import os
import sys
import random
from openai import AzureOpenAI
sys.path.append(os.getcwd())
from copy import deepcopy

from utils import read_josnl, save_arr

if __name__ == '__main__':

    summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', type=str, default='xsum')
    argparser.add_argument('--token_control', action="store_true")
    args = argparser.parse_args()

    test_path = './data/{}/test_random1.jsonl'.format(args.data_name)
    train_path = './data/{}/train_random1.jsonl'.format(args.data_name)
    client = AzureOpenAI()

    test_data = read_josnl(test_path)
    train_data = read_josnl(train_path)

    generate_result  = []

    summary_prompt = "\nSummarize the above article."
    if args.token_control:
        if args.data_name == "cnn_dailymail":
            summary_prompt = "\nSummarize the article in three sentences around 60 words."
        if args.data_name == "xsum":
            summary_prompt = "\nSummarize the article in one sentence around 35 words."
    print("summary ptompy", summary_prompt)

    system_prompt = """Generate a concise and coherent summary towards the given article and don't generate anything else. Make sure the summary is clear, informative, and well-structured.\n\n"""
    
    for instance in tqdm.tqdm(test_data):
        article_key = summarization_name_mapping[args.data_name][0]

        ## one-shot
        selected_example = random.choice(train_data)
        selected_article = selected_example[article_key]
        selected_words = selected_article.split(" ")
        selected_article = " ".join(selected_words[:2048])
        final_example = "Here is a example.\n\n\n\n Article: {article}. \n\n {dataset_specific_prompt} \n\n Summary: {summary} \n\n\n".format(article=selected_article, dataset_specific_prompt=summary_prompt, summary=selected_example["pseudo_summary"])


        source = instance[article_key]
        words = source.split(" ")
        article = " ".join(words[:2048])
        user_prompt = final_example + "Article: {article}. \n\n {dataset_specific_prompt} \n\n Summary:".format(article=article, dataset_specific_prompt=summary_prompt)
        while True:
            try:
                _response = client.chat.completions.create(
                    model="GPT-3-16k", # GPT-4-turbo / gpt-4, GPT-3-16k / gpt-35-turbo-16k
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_tokens=512,
                )
                time.sleep(0.5)
                resp = _response.choices[0].message.content.strip()
                new_source = deepcopy(instance)
                new_source["gpt35_turbo"] = resp 
                generate_result.append(new_source)
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                    break
                else:
                    print('ignored', instance)
                    break

    save_arr(generate_result, "./logs/results/gpt35/{}_oneshot.jsonl".format(args.data_name + str(args.token_control)))
    print("save success")
