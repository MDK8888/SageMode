import torch
import torchvision.models as models
from sagemode.Types.IO import *
from sagemode.ResourceUser.PyTorchEC2ResourceUser import PyTorchEC2ResourceUser
import base64

def save_resnet_18_weights_to_path(weight_path:str):
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), weight_path)

def pre_process(input:dict):
    from torchvision import transforms
    from PIL import Image
    import io
    import base64

    base64_string = input["base64"]
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

def post_process(output) -> dict:
    output = output.tolist()
    output = output[0]
    return {"logits":output, "parameters":None}

resnet18_local_weight_path = "resnet18_weights.pth"
save_resnet_18_weights_to_path(resnet18_local_weight_path)

user = PyTorchEC2ResourceUser("d2.2xlarge", IOTypes.ImageModeling)

ami_id = "ami-0ee4f2271a4df2d7d"
model_path = "model.py"
user.deploy(ami_id, 
            model_path, 
            resnet18_local_weight_path, 
            pre_process, 
            post_process, 
            "Resnet18Lambda", 
            ["python", "-m", "pip"], 
            "ec2_requirements.txt"
            )

image_path = "fish.jpg"

with open(image_path, "rb") as image_bytes:
    binary_data = image_bytes.read()
    base64_encoded = base64.b64encode(binary_data).decode('utf-8')

input = {"base64": base64_encoded, "parameters":None}

print(user.use(input))