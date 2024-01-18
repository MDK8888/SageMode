#In order for speculative decoding to work, the vocabulary for two models must be the same, i.e. the dictionary with id keys and string tokens must be the same.
import torch
import torch.nn as nn
from typing import Callable

#ok, here's the key behind speculative decoding. We have two models, Mq the small model and Mp the large model. 
#1. Run Mq on prefix and obtain the distribution for x1 q(x).
#2. Run Mp on prefix and prefix + x1 concurrently to get the distributions for x2 and p(x).
#3. If x1 is rejected by Mp, reject and resample x1 from an altered distribution, otherwise keep x1 and x2. 

def add_speculative_decode(model:nn.Module):

    def speculative_decode(
        self,
        draft_model:nn.Module,
        input_pos:int,
        x:torch.Tensor,
        speculate_k:int,
        model_decode_function:Callable,
        draft_model_decode_function:Callable,
        sampling_function:Callable,
        **sampling_kwargs
    ) -> torch.Tensor:
        device = x.device

        draft_model_sampling_kwargs = sampling_kwargs["draft_model_sampling_kwargs"]
        draft_tokens, draft_prob = draft_model_decode_function(**draft_model_sampling_kwargs)

        if type(draft_tokens) == list:
            draft_tokens = torch.cat(draft_tokens)
        
        model_sampling_kwargs = sampling_kwargs["model_sampling_kwargs"]
        model_tokens, model_prob = model_decode_function(x, draft_tokens, **model_sampling_kwargs)
        
        if type(model_tokens) == list:
            model_tokens = torch.cat(model_tokens)
        
        p = draft_prob[torch.arange(0, speculate_k, device=device), draft_tokens]
        q = model_prob[torch.arange(0, speculate_k, device=device), draft_tokens]

        accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
        rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

        if rejected_locations.shape[0] == 0:
            last_token = sampling_function(model_prob[-1])
            draft_model.forward(
                x=draft_tokens, 
                start_pos=input_pos
            )
            return torch.cat([draft_tokens, last_token])
        else:
            accept_length = rejected_locations[0].item()
            p = draft_prob[accept_length]
            q = model_prob[accept_length]
            new = q - p
            new = torch.where(new > 0, new, 0.0)
            next_token = sampling_function(new)
            return torch.cat([draft_tokens[:accept_length], next_token])
    
    model.speculative_decode = speculative_decode

    return model