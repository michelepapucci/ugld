# UGLD

Uncertainty-Gated Lexical Decoding (UGLD) logits processors for HuggingFace Transformers.

## Install

```bash
pip install ugld
```

## Usage
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from ugld import UGLD_Towards, UGLDTowardsConfig

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

simple_words = [
        " simple", " easy", " basic", " clear",
        " small", " big", " light", " heavy",
        " fast", " slow", " old", " new",
        " good", " bad", " near", " far",
        " start", " end", " help", " use",
    ]

green_ids = []
for w in simple_words:
    green_ids.extend(tok.encode(w, add_special_tokens=False))

green_ids = list(set(green_ids))

proc = LogitsProcessorList([
    UGLD_Towards(UGLDTowardsConfig(
        green_token_ids=green_ids,
        alpha_max=0.5,
        tau=1.0,
        s=0.3,
        prior="renorm",
    ))
])

inputs = tok("Explain gravity in", return_tensors="pt")

out = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=proc,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

You can find a notebook on how to start using UGLD in ```examples/quickstart.ipynb``` or hosted on [colab](https://colab.research.google.com/drive/1CyD7EESDZPpKaIYx7g_OQo6GEuEkkq2k?usp=sharing).