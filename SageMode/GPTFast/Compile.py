import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module, spec_decode:bool = False) -> nn.Module:
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    if spec_decode:
        assert hasattr(model, "draft_model"), "You have passed spec_decode = True in your torch_compile but your model doesn't have a draft model."
        

    return model