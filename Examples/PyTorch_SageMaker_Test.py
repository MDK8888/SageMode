import torch
import torchvision.models as models
from Types.IO import *
from ResourceUser.PyTorchSageMakerResourceUser import PyTorchSageMakerResourceUser
import base64

def save_resnet_18_weights_to_path(weight_path:str):
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), weight_path)

resnet18_local_weight_path = "resnet18_weights.pth"
save_resnet_18_weights_to_path(resnet18_local_weight_path)

user = PyTorchSageMakerResourceUser("ml.g5.xlarge", IOTypes.ImageModeling)

def model_fn(model_dir):
    from model import model
    import torch
    import os
    weight_path = os.path.join(model_dir, "weights.pth")
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    from torchvision import transforms
    from PIL import Image
    import io
    import json
    import base64

    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
        base64_string = request_body["base64"]
        img_data = base64.b64decode(base64_string)
        img_bytes = io.BytesIO(img_data)
        img = Image.open(img_bytes)

        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    import torch
    with torch.no_grad():
        predictions = model(input_data)
    return predictions

def output_fn(prediction, response_content_type):
    return {"logits": prediction.numpy().tolist(), "parameters":None}
    
functions_dict = {
    "model_fn": model_fn,
    "input_fn": input_fn,
    "predict_fn": predict_fn,
    "output_fn": output_fn
}
user.deploy("Resnet18_Lambda", functions_dict, "model.py", "resnet18_weights.pth", False, False, "3.8", {"python_version": "py38", "pytorch_version":"1.10"}, "sagemaker_requirements.txt")

image_path = "fish.jpg"

with open(image_path, "rb") as image_bytes:
    binary_data = image_bytes.read()
    base64_encoded = base64.b64encode(binary_data).decode('utf-8')

input = {"base64": base64_encoded, "parameters":None}

print(user.use(input))
