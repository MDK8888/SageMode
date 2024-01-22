import torch
import torch.nn as nn
import torch.quantization as quantization

def quantize_model(model:nn.Module, dtype='float32') -> nn.Module:
    # Ensure the model is in evaluation mode
    model.eval()

    # Specify the quantization configuration
    if dtype == 'float16':
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif dtype == 'int8':
        qconfig = torch.quantization.get_default_qconfig('qnnpack')
    else:
        # Default to float32
        return model

    # Quantize the model
    quantized_model = quantization.quantize_dynamic(
        model, qconfig_spec=qconfig, dtype=torch.__dict__[dtype]
    )

    return quantized_model
