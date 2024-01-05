from sagemode.ResourceUser.HFSageMakerResourceUser import HFSageMakerResourceUser

user = HFSageMakerResourceUser()
user.deploy("google/flan-t5-small", "FlanT5Lambda")

user.use({"text":"Hello, how are you?", "parameters": None})