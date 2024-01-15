from sagemode.ResourceUser import HFSageMakerResourceUser

user = HFSageMakerResourceUser()

user.deploy(model_id="google/flan-t5-small", instance_type="ml.g5.xlarge", skips=["compression", "download", "upload"], lambda_function_name="FlanT5Lambda")

response = user.use({"text": "Hello, how are you?", "parameters":None})

print("Flan-T5 response:", response)

user.teardown()