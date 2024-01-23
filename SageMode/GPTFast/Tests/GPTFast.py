from transformers import AutoModelForCausalLM
from ..GPTFast import gpt_fast

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

model = gpt_fast(model)