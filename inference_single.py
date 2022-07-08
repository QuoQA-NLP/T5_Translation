from transformers import AutoTokenizer, MarianMTModel, AutoTokenizer, AutoModelForSeq2SeqLM
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

# https://huggingface.co/datasets/conceptual_captions
src_text = [
    "sierra looked stunning in this top and this skirt while performing with person at their former university"
]

# model_name = "/home/ubuntu/En_to_Ko/ke-t5-base-finetuned-en-to-ko/checkpoint-17850"
model_name = CFG.inference_model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)

translated = model.generate(
    **tokenizer(src_text, return_tensors="pt", padding=True, max_length=CFG.max_token_length,),
    max_length=CFG.max_token_length,
    num_beams=CFG.num_beams,
    repetition_penalty=CFG.repetition_penalty,
    no_repeat_ngram_size=CFG.no_repeat_ngram_size,
    num_return_sequences=CFG.num_return_sequences,
)
print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
