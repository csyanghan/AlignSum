# AlignSum: Data Pyramid Hierarchical Fine-tuning for Aligning with Human Summarization Preference

## Data Preparation

You need to download the original [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum?row=0) and [CNNDM](https://huggingface.co/datasets/cnn_dailymail), put them in `data/original`

`element-aware-cnndm` and `element-aware-xsum` can be download from [https://github.com/Alsace08/SumCoT](https://github.com/Alsace08/SumCoT), and put them in `data/cnn_dailymail` and `data/xsum`.

## Element-Aware Dataset Sample and Data Pyramid Construction

```bash
# Select 100 samples as human-annotated data and another 100 samples as test data.
# You can also use the sampled data from our experiments located in the sampled-data directory.
python dp/sample_user_specific_data.py --dataset cnn_dailymail

python dp/generate_GSG.py --dataset cnn_dailymail
CUDA_VISIBLE_DEVICES=0 python dp/generate_LLM.py --dataset cnn_dailymail
python dp/merge_and_gaussian.py --dataset cnn_dailymail
```

## Two-stage Hierarchical Fine-tuning

```bash
bash scripts/cnn_ft_stage1.sh
bash scripts/cnn_ft_stage2.sh
```

## Automatic Evaluation

```bash
python bs_score.py
```
You may need to change the result path.
