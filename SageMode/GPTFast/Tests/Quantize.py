from transformers import AutoModelForCausalLM
from ..Quantize import *

model_name = "gpt2-xl"

model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model.cuda()

model = load_from_int8(model)