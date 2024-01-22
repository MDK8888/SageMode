import time
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ..SpeculativeDecode import speculative_decode

def generate_probability_distribution(initial_token_ids, length, model, callback, return_text: bool = True, **kwargs):
    # Encode the initial token
    input_ids = initial_token_ids.unsqueeze(0)

    all_probabilities = []

    for _ in range(length):
        # Extract the logits from the output
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            # Get the tokens and their probabilities as a tensor
            token_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Use the callback function for token sampling, passing any additional kwargs
        next_token_id = callback(token_probabilities, **kwargs)

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

# Example usage
model_name = "gpt2-xl"
draft_model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

initial_string = "Hello, how are you?"
input_tokens = tokenizer.encode(initial_string, return_tensors="pt").squeeze(0)

def argmax(probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index

model.speculative_decode = types.MethodType(speculative_decode, model)

t1 = time.time()

model_tokens = model.speculative_decode(
    draft_model, 
    input_tokens, 
    20, 
    generate_probability_distribution, 
    argmax,
    False,
    draft_model_decoding_kwargs={"model":draft_model, "callback":argmax, "return_text":True}
)

print(f"speculative decode time: {time.time() - t1:.2f}")

print(model_tokens)

model_tokens = model_tokens.long().tolist()

print(tokenizer.decode(model_tokens, skip_special_tokens=True))

t1 = time.time()

original_tokens, original_prob = generate_probability_distribution(input_tokens, 20, model, argmax)

print(f"normal inference time: {time.time() - t1:.2f}")