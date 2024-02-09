import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module, spec_decode:bool = False) -> nn.Module:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    if spec_decode:

        assert hasattr(model, "draft_model"), "You have passed spec_decode = True in your torch_compile but your model doesn't have a draft model."
        draft_model = model.draft_model
        draft_model = draft_model.to(device)
        draft_model = torch.compile(draft_model, mode="reduce-overhead", fullgraph=True)

    return model