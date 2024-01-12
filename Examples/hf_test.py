from sagemode.ResourceUser.HFSageMakerResourceUser import HFSageMakerResourceUser

user = HFSageMakerResourceUser("ml.g5.xlarge")
user.deploy("google/flan-t5-small", "FlanT5Lambda")

user.use({"text":"Hello, how are you?", "parameters": None})