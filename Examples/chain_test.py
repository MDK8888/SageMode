import base64
from sagemode.Types.Arn import *
from sagemode.ResourceUser.HFSageMakerResourceUser import *
from sagemode.DeploymentStateMachine.DeploymentStateMachine import *
from sagemode.Types.IO import IOTypes

users = [None, None]
users[0] = HFSageMakerResourceUser("ml.g5.xlarge", IOTypes.LanguageModeling, IOTypes.LanguageModeling)
users[1] = HFSageMakerResourceUser("ml.g5.xlarge", IOTypes.LanguageModeling, IOTypes.ImageModeling)

deployment_args = [None, None]
deployment_args[0] = {"model_id": "google/flan-t5-small", "function_name":"FlanT5SmallLambda", "skip_compression":True, "skip_upload":True}
deployment_args[1] = {"model_id": "OFA-Sys/small-stable-diffusion-v0", "function_name": "SmallStableDiffusionV0Lambda", "timeout":240, "skip_compression":True, "skip_upload":True}

deployment_state_machine = DeploymentStateMachine()
deployment_state_machine.deploy("FirstStateMachine", None, users, deployment_args)

result = deployment_state_machine.use({"text": "It was a dark and stormy night...", "parameters": None}, 5)

base64_encoded = result["base64"]

decoded_data = base64.b64decode(base64_encoded)

with open("output.png", 'wb') as output_file:
    output_file.write(decoded_data)