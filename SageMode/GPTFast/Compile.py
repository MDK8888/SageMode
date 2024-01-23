import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module, spec_decode:bool = False) -> nn.Module:
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    if spec_decode:
        assert hasattr(model, "speculative_decode"), "You must have a speculative decode function in your model."
        model.speculative_decode = torch.compile(model.speculative_decode, mode="reduce-overhead", fullgraph=True)
        
        assert hasattr(model, "generate"), "You must have a generate function in your model."
        model.generate = torch.compile(model.generate, mode="reduce-overhead", fullgraph=True)

        assert hasattr(model, "draft_model"), "You have passed spec_decode = True in your torch_compile but your model doesn't have a draft model."
        draft_model = model.draft_model
        draft_model.forward = torch.compile(draft_model.forward, mode="reduce-overhead", fullgraph=True)
        
        assert hasattr(draft_model, "decode_function"), "Your draft model must have a decode function."
        draft_model.decode_function = torch.compile(draft_model.decode_function, mode="reduce-overhead", fullgraph=True)

    return model