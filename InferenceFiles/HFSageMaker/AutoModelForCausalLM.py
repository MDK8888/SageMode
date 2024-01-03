import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_fn(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)

    input_tokens = tokenizer(inputs, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input_tokens)
    logits = outputs.logits
    prediction = tokenizer.decode(logits[0], skip_special_tokens=True)
    return {"text":prediction, "parameters":None}
