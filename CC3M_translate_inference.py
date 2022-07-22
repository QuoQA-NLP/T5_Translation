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
    T5Tokenizer
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
CFG_dset_name = "conceptual_captions"
train_dataset = load_dataset(CFG.dset_name, split="train")
valid_dataset = load_dataset(CFG.dset_name, split="validation")
print(train_dataset)
print(valid_dataset)


tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
model.to(device)


start = 0
batch_size = 600 # P100:batch_size 250 / A100:batch_size 600
length = len(train_dataset)
cnt =  length//batch_size + 1
df = pd.DataFrame(columns = {"english_caption", "korean_caption", "image_url"})

# start train dastasets translate

csv_start = 0
save_start = csv_start
save_count = 0
for i in tqdm(range(start,cnt)):
  save_count+=1
  check = False
  
  end=csv_start+batch_size
  if end>len(train_dataset):
    check = True
    end = len(train_dataset)
  
  src_sentences = train_dataset['caption'][csv_start:end]
  urls = train_dataset['image_url'][csv_start:end]

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

  df1 = pd.DataFrame({"english_caption": src_sentences, "korean_caption": generated_texts, "image_url": urls})
  df = df.append(df1, ignore_index = True)
  if save_count == 30 or check==True:
    save_count=0
    df.to_csv(f"./results/train_translated-{save_start}-{end}-sentences.csv", index=False)
  csv_start = end


start = 0
batch_size = 600 # P100:batch_size 250 / A100:batch_size 600
length = len(valid_dataset)
cnt =  length//batch_size + 1
df = pd.DataFrame(columns = {"english_caption", "korean_caption", "image_url"})

# start validation dastasets translate

csv_start = 0
save_start = csv_start
save_count = 0
for i in tqdm(range(start,cnt)):
  save_count+=1
  check = False
  
  end=csv_start+batch_size
  if end>len(valid_dataset):
    check = True
    end = len(valid_dataset)
  
  src_sentences = valid_dataset['caption'][csv_start:end]
  urls = valid_dataset['image_url'][csv_start:end]

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

  df1 = pd.DataFrame({"english_caption": src_sentences, "korean_caption": generated_texts, "image_url": urls})
  df = df.append(df1, ignore_index = True)
  if save_count == 30 or check==True:
    save_count=0
    df.to_csv(f"./results/valid_translated-{save_start}-{end}-sentences.csv", index=False)
  csv_start = end




# df = pd.DataFrame({"src": src_sentences, "tgt": tgt_sentences, "gen": generated_texts})
# df.to_csv(f"./results/translated-{CFG.no_inference_sentences}-sentences.csv", index=False)