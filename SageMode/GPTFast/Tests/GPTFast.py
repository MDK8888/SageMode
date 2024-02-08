import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..GPTFast import gpt_fast

torch._dynamo.reset()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def generate_probability_distribution_static(model, input_ids, length, return_text:bool = True):
    # Encode the initial token

    all_probabilities = []

    for _ in range(length):
        # Extract the logits from the output
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            # Get the tokens and their probabilities as a tensor
            token_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Use the callback function for token sampling, passing any additional kwargs
        max_prob_index = torch.argmax(token_probabilities, dim=-1)
        next_token_id = max_prob_index

        # Append the sampled token to the input sequence
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

        # Append the probabilities to the list
        all_probabilities.append(token_probabilities.unsqueeze(0))

    # Stack the probabilities to create a tensor of size (length, vocab_size)
    all_probabilities_tensor = torch.cat(all_probabilities, dim=0)

    if return_text:
        return input_ids.squeeze(0)[-length:], all_probabilities_tensor.squeeze(1)
    else:
        return all_probabilities_tensor.squeeze(1)

def generate_probability_distribution(self, input_ids, length, return_text:bool = True):
    # Encode the initial token

    all_probabilities = []

    for _ in range(length):
        # Extract the logits from the output
        with torch.no_grad():
            logits = self.forward(input_ids).logits[:, -1, :]
            # Get the tokens and their probabilities as a tensor
            token_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Use the callback function for token sampling, passing any additional kwargs
        max_prob_index = torch.argmax(token_probabilities, dim=-1)
        next_token_id = max_prob_index

        # Append the sampled token to the input sequence
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(1)], dim=-1)

        # Append the probabilities to the list
        all_probabilities.append(token_probabilities.unsqueeze(0))

    # Stack the probabilities to create a tensor of size (length, vocab_size)
    all_probabilities_tensor = torch.cat(all_probabilities, dim=0)

    if return_text:
        return input_ids[:, -length:], all_probabilities_tensor.squeeze(1)
    else:
        return all_probabilities_tensor.squeeze(1)

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index

model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

draft_model_name = "gpt2"
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").to(device)

N_ITERS=10
MAX_TOKENS=50

'''
model.to(device)

compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: generate_probability_distribution_static(model, input_tokens, MAX_TOKENS, True))
    compile_times.append(compile_time)
    print(f"eager eval time {i}: {compile_time}")

model.to("cpu")
'''

gpt_fast_model = gpt_fast(model, draft_model=draft_model, draft_model_decode_function=generate_probability_distribution, sample_function=argmax)
gpt_fast_model.to(device)

fast_compile_times = []
for i in range(N_ITERS):
    with torch.no_grad():
        _, compile_time = timed(lambda: gpt_fast_model.generate(cur_tokens=input_tokens, max_tokens=MAX_TOKENS, speculate_k=5, draft_model_decoding_kwargs={"return_text":True}))
    fast_compile_times.append(compile_time)
    print(f"gpt fast eval time {i}: {compile_time}")
print("~" * 10)