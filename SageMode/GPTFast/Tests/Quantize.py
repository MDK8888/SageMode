import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..Quantize import quantize_model

model_name = "gpt2-xl"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval().cuda()

input = "Hello, how are you today?"
input_tokens = tokenizer.encode(input, return_tensors="pt")

t0 = time.time()
output = model(input_tokens)
print(f"time taken: {time.time() - t0:.2f}")

model = quantize_model(model, "int8")

t0 = time.time()
output = model(input_tokens)
print(f"time taken: {time.time() - t0:.2f}")