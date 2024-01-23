import time
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ..SpeculativeDecode import add_speculative_decoding

def generate_probability_distribution(self, input_ids, length, return_text:bool = True, **kwargs):
    # Encode the initial token

    all_probabilities = []

    for _ in range(length):
        # Extract the logits from the output
        with torch.no_grad():
            logits = self.forward(input_ids).logits[:, -1, :]
            # Get the tokens and their probabilities as a tensor
            token_probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Use the callback function for token sampling, passing any additional kwargs
        next_token_id = argmax(token_probabilities, **kwargs)

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

def argmax(probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index

# Example usage
model_name = "gpt2-xl"
draft_model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

add_speculative_decoding(model, draft_model, generate_probability_distribution, argmax)




