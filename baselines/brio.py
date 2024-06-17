import os
import sys
import torch
sys.path.append(os.getcwd())
from tqdm import tqdm
from utils import read_josnl, save_arr

from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

summarization_name_mapping = {
        "cnn_dailymail": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }
dataset = "xsum"

test_dataset = read_josnl("data/{}/test_random1.jsonl".format(dataset))
device = torch.device("cuda:2")


IS_CNNDM = dataset == "cnn_dailymail" # whether to use CNNDM dataset or XSum dataset

# Load our model checkpoints
if IS_CNNDM:
    model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
    tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
else:
    model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
    tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

model = model.to(device)

max_length = 1024 if IS_CNNDM else 512
# generation example

results = []
for sample in tqdm(test_dataset):
    article_key = summarization_name_mapping[dataset][0]
    article = sample[article_key]
    inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"].to(device), max_length=128)
    pred = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    sample["brio_pred"] = pred
    results.append(sample)

save_arr(results, "logs/results/brio/{}.jsonl".format(dataset))
