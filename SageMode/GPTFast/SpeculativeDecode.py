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
    decode_function_str:Callable,
    sampling_function_str:Callable,
    kv_cache:bool = False,
    **kwargs
) -> torch.Tensor:
    device = cur_tokens.device

    draft_model_sampling_kwargs = kwargs["draft_model_decoding_kwargs"]

    if kv_cache:
        decode_input = torch.Tensor([cur_tokens[-1]]) #the KV cache if implemented correctly only needs the most recent token.
    else:
        decode_input = cur_tokens

    assert hasattr(self, "draft_model"), "You did not prepare your model properly for speculative decoding. Make sure that you add a draft model."
    draft_model = self.draft_model

    draft_model_decode_function = getattr(draft_model, decode_function_str)
    sampling_function = getattr(self, sampling_function_str)
    
    draft_tokens, draft_prob = draft_model_decode_function(decode_input, speculate_k, **draft_model_sampling_kwargs)

    assert len(draft_tokens.shape) == 1 and len(draft_prob.shape) == 2, "Your draft tokens must have shape (seq_len) and draft_prob must have shape (seq_len, vocab_size)."

    if type(draft_tokens) == list:
        draft_tokens = torch.cat(draft_tokens)
    
    model_sampling_kwargs = kwargs.get("model_forward_kwargs", {})
    model_logits = self.forward(draft_tokens, **model_sampling_kwargs).logits.squeeze(0)
    model_prob = torch.nn.functional.softmax(model_logits, dim=-1)
    
    assert len(model_prob.shape) == 2, "Your model_prob must have shape (seq_len, vocab_size)."

    assert len(model_prob) == len(draft_prob), "In order for speculative decoding to work, the main model must the same number of tokens as the draft model."

    p = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

    ratio = p / q
    rand = torch.rand_like(ratio)
    n = torch.argmax((rand > ratio).to(dtype=torch.float32)).item()

    if n < len(ratio) and rand[n] > ratio[n]:
        p = draft_prob[n]
        q = model_prob[n]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        last_token = torch.Tensor([sampling_function(new)])
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

def generate(self, cur_tokens:torch.Tensor, max_tokens:int, speculate_k:int, decode_function_str:str, sampling_function_str:str, **kwargs) -> torch.Tensor:

    assert len(cur_tokens.shape) == 2 and cur_tokens.shape[0] == 1, "Your batch size must be 1"

    assert hasattr(self, "speculative_decode"), "You must attach speculative decoding as a method of the LLM"

    while len(cur_tokens[0]) < max_tokens:
        new_tokens = self.speculative_decode(cur_tokens, speculate_k, decode_function_str, sampling_function_str, **kwargs)
        cur_tokens[0] = torch.cat((cur_tokens[0], new_tokens), dim=0)

    return cur_tokens

def add_speculative_decoding(model:nn.Module, draft_model:nn.Module, draft_model_decode_function:Callable, sample_function:Callable) -> nn.Module:
    draft_model_decode_function_name = draft_model_decode_function.__name__
    setattr(draft_model, draft_model_decode_function_name, draft_model_decode_function)
    model.draft_model = draft_model

    sample_function_name = sample_function.__name__
    setattr(model, sample_function_name, sample_function)

    model.speculative_decode = types.MethodType(speculative_decode, model)    
    model.generate = types.MethodType(generate, model)
    return model




    
    
    

    




