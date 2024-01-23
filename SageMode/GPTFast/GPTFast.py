import os
from torch import nn
from .Compile import torch_compile_model
from .Quantize import *
from .SpeculativeDecode import speculative_decode

def gpt_fast(model:nn.Module):
    int8_path = f"{os.getcwd()}/{model.config.name_or_path}int8.pth"
    if not os.path.exists(int8_path):
        int8_quantize(model)
    model = load_from_int8(model)
    model = torch_compile_model(model)
    return model
