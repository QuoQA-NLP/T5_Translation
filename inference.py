import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    MarianMTModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
)
import multiprocessing
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = CFG.inference_model_name
valid_dataset = load_dataset(CFG.dset_name, split="valid", use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
model.to(device)

CPU_COUNT = multiprocessing.cpu_count() // 2
selected_valid = valid_dataset.select(range(0, CFG.no_inference_sentences))
src_sentences = selected_valid[CFG.src_language]
tgt_sentences = selected_valid[CFG.tgt_language]

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

# https://github.com/huggingface/transformers/issues/10704
generated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(generated_texts)

df = pd.DataFrame({"src": src_sentences, "tgt": tgt_sentences, "gen": generated_texts})
df.to_csv(f"./results/translated-{CFG.no_inference_sentences}-sentences.csv", index=False)
