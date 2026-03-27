# UGLD Documentation

Uncertainty-Gated Lexical Decoding (UGLD) for HuggingFace Transformers.

## Installation

```bash
pip install ugld
```

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from ugld import UGLD_Towards, UGLDTowardsConfig

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

simple_words = [" simple", " easy", " basic", " clear"]

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

## API docs

Full API reference is available in [API Reference](api.md).

The API docs are generated from code docstrings and published to GitHub Pages via workflow.

- `ugld.UGLD_Towards`
- `ugld.UGLD_Against`
- `ugld.UGLDTowardsConfig`
- `ugld.UGLDAgainstConfig`

## Local preview

```bash
pdoc --http 8000 ugld
```

Then open http://127.0.0.1:8000/ugld 
