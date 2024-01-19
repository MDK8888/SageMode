from torch import nn
from Compile import torch_compile_model
from KVCache import modify_transformer_attention_blocks
from Quantize import quantize_model
from SpeculativeDecode import speculative_decode

def gpt_fast(model:nn.Module, **transformer_kwargs):
    model = torch_compile_model(model)
    model = modify_transformer_attention_blocks(model, **transformer_kwargs)
    model = quantize_model(model)
    model.speculative_decode = speculative_decode
    return model
