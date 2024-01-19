#In order for speculative decoding to work, the vocabulary for two models must be the same, i.e. the dictionary with id keys and string tokens must be the same.
import torch
import torch.nn as nn
from typing import Callable

#ok, here's the key behind speculative decoding. We have two models, Mq the small model and Mp the large model. 
#1. Run Mq on prefix and obtain the distribution for x1 q(x).
#2. Run Mp on prefix and prefix + x1 concurrently to get the distributions for x2 and p(x).
#3. If x1 is rejected by Mp, reject and resample x1 from an altered distribution, otherwise keep x1 and x2. 

def speculative_decode(
    self,
    draft_model:nn.Module,
    cur_tokens:torch.Tensor,
    speculate_k:int,
    model_decode_function:Callable,
    draft_model_decode_function:Callable,
    sampling_function:Callable,
    kv_cache:bool = False,
    **kwargs
) -> torch.Tensor:
    device = cur_tokens.device

    draft_model_sampling_kwargs = kwargs["draft_model_decoding_kwargs"]

    if kv_cache:
        decode_input = torch.Tensor([cur_tokens[-1]]) #the KV cache if implemented correctly only needs the most recent token.
    else:
        decode_input = cur_tokens

        draft_tokens, draft_prob = draft_model_decode_function(decode_input, speculate_k, **draft_model_sampling_kwargs)

    if type(draft_tokens) == list:
        draft_tokens = torch.cat(draft_tokens)
    
    model_sampling_kwargs = kwargs["model_decoding_kwargs"]
    model_prob = model_decode_function(decode_input, speculate_k+1, **model_sampling_kwargs)

    assert len(model_prob) == len(draft_prob) + 1, "In order for speculative decoding to work, the main model must sample one more token than the draft model."
    
    p = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

    ratio = p / q
    rand = torch.rand_like(ratio)
    n = torch.argmax(rand > ratio).item()

    if n < len(ratio) and rand[n] > ratio[n]:
        p = draft_prob[n]
        q = model_prob[n]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        last_token = sampling_function(new)
        if kv_cache:
            assert hasattr(self, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            self.rollback_cache(n + len(cur_tokens) + 1)
            assert hasattr(draft_model, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            draft_model.rollback_cache(n + len(cur_tokens) + 1)
        
        return torch.cat([draft_tokens[:n+1], last_token])
    else: #we accept all tokens from the draft model
        last_token = sampling_function(model_prob[-1])
        if kv_cache:
            assert hasattr(self, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            self.rollback_cache(n + len(cur_tokens) + 2)
            assert hasattr(draft_model, "rollback_cache"), "Error: In order for speculative decoding to work with a kv cache, you must be able to update it."
            draft_model.rollback_cache(n + len(cur_tokens) + 1)

        #assume that draft_model already has a kv cache attached.
        return torch.cat([draft_tokens, last_token])