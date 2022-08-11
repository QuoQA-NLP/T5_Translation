# T5 Machine Translation: English ↔️ Korean

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/QuoQA-NLP/QuoQaGo)

### Result

|                   | BLEU Score |                                                Translation Result                                                |
| :---------------: | :--------: | :--------------------------------------------------------------------------------------------------------------: |
| English ➡️ Korean |   45.148   | [KE-T5-Ko2En-Base Inference Result](https://huggingface.co/datasets/QuoQA-NLP/KE-T5-Ko2En-Base-Inference-Result) |
| Korean ➡️ English |     -      |                                                                                                                  |

- Evaluation script is on [metric.py](./metric.py)
- English ➡️ Korean Result evaluated on 553500 sentence pairs which are disjoint from the train set.

### How to Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Korean -> English Machine Translation
tokenizer = AutoTokenizer.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")
model = AutoModelForSeq2SeqLM.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")

# English -> Korean Machine Translation
tokenizer = AutoTokenizer.from_pretrained("QuoQA-NLP/KE-T5-En2Ko-Base")
model = AutoModelForSeq2SeqLM.from_pretrained("QuoQA-NLP/KE-T5-En2Ko-Base")
```

- For batch translation, please refer to [inference.py](./inference.py).
  - P100 16GB supports inferencing of 250 pairs per batch on device.
  - A100 40GB supports inferencing of 600 pairs per batch on device.
- For single sentence translation, please refer to [inference_single.py](./inference_single.py).

### References

- [🔗 Dataset specification](https://github.com/snoop2head/Deep-Encoder-Shallow-Decoder#dataset)
- [Translation Example](https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb)
- [Summarization Example](https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb)
- [Deep Encoder Shallow Decoder](https://github.com/snoop2head/Deep-Encoder-Shallow-Decoder/blob/main/trainer.py)
