#In order for speculative decoding to work, the vocabulary for two models must be the same, i.e. the dictionary with id keys and string tokens must be the same.
import types
import torch
import torch.nn as nn
from typing import Callable

#ok, here's the key behind speculative decoding. We have two models, Mq the small model and Mp the large model. 
#1. Run Mq on prefix and obtain the distribution for x1 q(x).
#2. Run Mp on prefix and prefix + x1 concurrently to get the distributions for x2 and p(x).
#3. If x1 is rejected by Mp, reject and resample x1 from an altered distribution, otherwise keep x1 and x2. 

def speculative_decode(
    self,
    cur_tokens:torch.Tensor,
    speculate_k:int,
    kv_cache:bool = False,
    **kwargs
) -> torch.Tensor:

    device = cur_tokens.device

    draft_model_sampling_kwargs = kwargs.get("draft_model_decoding_kwargs", {})

    if kv_cache:
        decode_input = torch.Tensor([cur_tokens[-1]]) #the KV cache if implemented correctly only needs the most recent token.
    else:
        decode_input = cur_tokens

    assert hasattr(self, "draft_model"), "You did not prepare your model properly for speculative decoding. Make sure that you add a draft model."
    draft_model = self.draft_model
    
    draft_tokens, draft_prob = draft_model.decode_function(input_ids=decode_input, length=speculate_k, **draft_model_sampling_kwargs)

    assert len(draft_tokens.shape) == 2 and len(draft_prob.shape) == 2, "Your draft tokens must have shape (1, seq_len) and draft_prob must have shape (seq_len, vocab_size)."

    model_sampling_kwargs = kwargs.get("model_forward_kwargs", {})
    full_tokens = torch.cat([decode_input, draft_tokens], dim=-1).to(device)
    with torch.no_grad():
        model_logits = self.forward(full_tokens, **model_sampling_kwargs).logits
    model_logits = model_logits.squeeze(0)[-draft_tokens.shape[1]:, :]
    model_prob = torch.nn.functional.softmax(model_logits, dim=-1)
    
    assert len(model_prob.shape) == 2, "Your model_prob must have shape (seq_len, vocab_size)."

    assert len(model_prob) == len(draft_prob), "In order for speculative decoding to work, the main model must generate the same number of tokens as the draft model."

    p = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

    ratio = p / q
    rand = torch.rand_like(ratio)

    n = (rand > ratio).nonzero(as_tuple=True)[0][0].item()

    if n < len(ratio) and rand[0][n] > ratio[0][n]:
        p = draft_prob[n]
        q = model_prob[n]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        last_token = torch.Tensor([self.sample(new)]).to(device)
        if kv_cache:
            assert hasattr(self, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            self.rollback_cache(n + len(cur_tokens) + 1)
            assert hasattr(draft_model, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            draft_model.rollback_cache(n + len(cur_tokens) + 1)
        
        return torch.cat([draft_tokens[:n+1], last_token.unsqueeze(0)], dim=-1).long()
    else: #we accept all tokens from the draft model
        last_token = torch.Tensor([self.sample(model_prob[-1])]).to(device)
        if kv_cache:
            assert hasattr(self, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            self.rollback_cache(n + len(cur_tokens) + 2)
            assert hasattr(draft_model, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            draft_model.rollback_cache(n + len(cur_tokens) + 1)

        #assume that draft_model already has a kv cache attached.
        return torch.cat([draft_tokens, last_token.unsqueeze(0)], dim=-1).long()

def generate(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, **kwargs) -> torch.Tensor:

    assert len(cur_tokens.shape) == 2 and cur_tokens.shape[0] == 1, "Your batch size must be 1"

    assert hasattr(self, "speculative_decode"), "You must attach speculative decoding as a method of the model"

    while len(cur_tokens[0]) < max_tokens:
        new_tokens = self.speculative_decode(cur_tokens, speculate_k, False, **kwargs)
        cur_tokens = torch.cat((cur_tokens, new_tokens), dim=1).to(torch.long)

    return cur_tokens

def add_speculative_decoding(model:nn.Module, draft_model:nn.Module, draft_model_decode_function:Callable, sample_function:Callable) -> nn.Module:
    draft_model.decode_function = types.MethodType(draft_model_decode_function, draft_model)
    model.draft_model = draft_model

    model.sample = types.MethodType(sample_function, model)

    model.speculative_decode = types.MethodType(speculative_decode, model)    
    model.generate = types.MethodType(generate, model)
    return model




    
    
    

    




