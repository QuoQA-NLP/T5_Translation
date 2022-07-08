from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import wandb
import numpy as np
from datasets import load_dataset, load_metric
import multiprocessing
from easydict import EasyDict
import yaml

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
    CFG = EasyDict(SAVED_CFG["CFG"])

metric = load_metric("sacrebleu")

# all dataset
dset = load_dataset(CFG.dset_name, use_auth_token=True)
tokenizer = T5Tokenizer.from_pretrained(CFG.model_name)  # https://github.com/AIRC-KETI/ke-t5#models


def preprocess_function(examples):
    inputs = examples[CFG.src_language]
    targets = examples[CFG.tgt_language]
    model_inputs = tokenizer(inputs, max_length=CFG.max_token_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=CFG.max_token_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# print(preprocess_function(dset["train"].select(range(0, 2))))

CPU_COUNT = multiprocessing.cpu_count() // 2

tokenized_datasets = dset.map(preprocess_function, batched=True, num_proc=CPU_COUNT)
tokenized_datasets


model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

str_model_name = CFG.model_name.split("/")[-1]
run_name = f"{str_model_name}-finetuned-{CFG.src_language}-to-{CFG.tgt_language}"
wandb.init(entity=CFG.entity_name, project=CFG.project_name, name=run_name)

training_args = Seq2SeqTrainingArguments(
    run_name,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.valid_batch_size,
    evaluation_strategy="steps",
    eval_steps=CFG.eval_steps,
    save_steps=CFG.save_steps,
    num_train_epochs=CFG.num_epochs,
    save_total_limit=CFG.num_checkpoints,
    predict_with_generate=True,
    fp16=CFG.fp16,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
)

wandb.config.update(training_args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()
trainer.save_model(CFG.save_path)
wandb.finish()
