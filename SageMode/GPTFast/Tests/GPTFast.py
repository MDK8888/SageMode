import torch
from transformers import AutoModelForCausalLM
from ..GPTFast import gpt_fast

model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name)

draft_model_name = "gpt2"
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

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
        return input_ids.squeeze(0)[-length:], all_probabilities_tensor.squeeze(1)
    else:
        return all_probabilities_tensor.squeeze(1)

def argmax(self, probabilities):
    # Use argmax to get the token with the maximum probability
    max_prob_index = torch.argmax(probabilities, dim=-1)
    return max_prob_index

model = gpt_fast(model, draft_model=draft_model, draft_model_decode_function=generate_probability_distribution, sample_function=argmax)