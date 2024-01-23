import os
from torch import nn
from .Compile import torch_compile_model
from .Quantize import *
from .SpeculativeDecode import add_speculative_decoding

def gpt_fast(model:nn.Module, **spec_dec_kwargs):
    int8_path = f"{os.getcwd()}/{model.config.name_or_path}int8.pth"
    if not os.path.exists(int8_path):
        int8_quantize(model)
    model = load_from_int8(model)
    spec_decode = False
    if spec_dec_kwargs:
        model = add_speculative_decoding(model, **spec_dec_kwargs)
        spec_decode = True
    model = torch_compile_model(model, spec_decode=spec_decode)
    return model
