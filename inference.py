from transformers import AutoTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

src_text = ["My name is Sarah and I live in London"]

model_name = "/home/ubuntu/En_to_Ko/ke-t5-base-finetuned-en-to-ko/checkpoint-1650"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translated = model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True, max_length=CFG.max_token_length,),
    max_length=CFG.max_token_length,
    no_repeat_ngram_size=1,
)
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
