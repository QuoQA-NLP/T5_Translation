import torch
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoTokenizer,
    MarianMTModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import multiprocessing
from easydict import EasyDict
import yaml


# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])
device = "cuda:0" if torch.cuda.is_available() else "cpu"

training_args = Seq2SeqTrainingArguments

model_name = CFG.inference_model_name
valid_dataset = load_dataset(CFG.dset_name, split="valid", use_auth_token=True)
print(valid_dataset)


tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
model.to(device)


start = 0
batch_size = 150  # P100:batch_size 250 / A100:batch_size 700
length = len(valid_dataset)
cnt = length // batch_size + 1
df = pd.DataFrame(columns={"src", "gen", "label"})

csv_start = 0
save_start = csv_start
save_count = 0
for i in tqdm(range(start, cnt)):
    save_count += 1
    if i == cnt - 1:
        end = len(valid_dataset)
    else:
        end = csv_start + batch_size

    src_sentences = valid_dataset["ko"][csv_start:end]
    label = valid_dataset["en"][csv_start:end]

    encoding = tokenizer(
        src_sentences, padding=True, return_tensors="pt", max_length=CFG.max_token_length
    ).to(device)

    # https://huggingface.co/docs/transformers/internal/generation_utils
    with torch.no_grad():
        translated = model.generate(
            **encoding,
            max_length=CFG.max_token_length,
            num_beams=CFG.num_beams,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            num_return_sequences=CFG.num_return_sequences,
        )
        del encoding

        # https://github.com/huggingface/transformers/issues/10704
        generated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        del translated
        print(generated_texts[0:2])

    df1 = pd.DataFrame({"src": src_sentences, "gen": generated_texts, "label": label})
    df = df.append(df1, ignore_index=True)
    if save_count == 30:
        save_count = 0
        df.to_csv(f"./results/tmp_translated-{save_start}-{end}-sentences.csv", index=False)
    csv_start = end

# load sacrebleu
# https://huggingface.co/spaces/evaluate-metric/sacrebleu | https://github.com/mjpost/sacreBLEU
metric = load_metric("sacrebleu")

preds = df["gen"]
labels = np.expand_dims(df["label"], axis=1)

score = metric.compute(predictions=preds, references=labels)  # takes 3 minutes for 550K pairs
print(score)

""" 
# Result of Korean to English Translation
{
    "score": 45.14821527744787,
    "counts": [10287887, 6969037, 5035938, 3719578],
    "totals": [14100267, 13546767, 12993267, 12439767],
    "precisions": [72.96235596106088, 51.44428187182964, 38.75805830819916, 29.90070473184908],
    "bp": 0.9886003016662179,
    "sys_len": 14100267,
    "ref_len": 14261929,
}
 """
