import torch
import torch.nn as nn

def torch_compile_model(model:nn.Module) -> nn.Module:
    model = torch.compile(model)
    return model

