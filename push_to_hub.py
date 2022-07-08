from transformers import AutoTokenizer, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

model_name = "/home/ubuntu/En_to_Ko/ke-t5-base-finetuned-en-to-ko/checkpoint-17850"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

config.push_to_hub("QuoQA-NLP/KE-T5-En2Ko-Base", private=True, use_temp_dir=True)
tokenizer.push_to_hub("QuoQA-NLP/KE-T5-En2Ko-Base", private=True, use_temp_dir=True)
model.push_to_hub("QuoQA-NLP/KE-T5-En2Ko-Base", private=True, use_temp_dir=True)
