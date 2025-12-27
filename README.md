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

green_ids = [tok.encode(" simple", add_special_tokens=False)[0]]

proc = LogitsProcessorList([
    UGLD_Towards(UGLDTowardsConfig(
        green_token_ids=green_ids,
        alpha_max=0.5,
        tau=3.0,
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
