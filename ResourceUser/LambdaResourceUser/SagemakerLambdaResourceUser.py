import os
import time
import zipfile
import json
from dotenv import load_dotenv
from ..ResourceUser import ResourceUser
from Types.Arn import *
from botocore.exceptions import ClientError

class SageMakerLambdaResourceUser(ResourceUser):

    def __init__(self, function_arn:LambdaArn=None):
        role_arn = RoleArn(os.environ["LAMBDA_ROLE_ARN"])
        super().__init__(role_arn)
        self.function_arn = function_arn
        self.lambda_client = self.boto3_session.client("lambda")
    
    def zip_lambda_file(self, file_name:str):
        local_zipped_lambda_dir = "lambda_zipped"
        with zipfile.ZipFile(f"{local_zipped_lambda_dir}.zip", 'w') as zipped:
            zipped.write(f"{file_name}.py")
            zipped.close()

    def deploy(self, function_name:str, endpoint_name:str=None, python_version:str="3.8", timeout:int=3) -> LambdaArn:
        if self.function_arn:
            raise ValueError("This object cannot call 'deploy' if it already has a function_arn - set 'self.function_arn = None' and try again.")
        
        lambda_function_file_path = "LambdaFunctions/Sagemaker/Sagemaker"
        self.zip_lambda_file(lambda_function_file_path)
        if endpoint_name is None:
            endpoint_name = os.environ["ENDPOINT_NAME"]

        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime=f'python{python_version}',
            Role=self.role_arn.raw_str,  # Replace with your Lambda execution role ARN
            Handler='LambdaFunctions/Sagemaker/Sagemaker.lambda_handler',  # Specify the module and function name
            Timeout=timeout,
            Code={
                'ZipFile': open(f"lambda_zipped.zip", "rb").read()  # Read the content of the file
            },
            Environment={
                'Variables': {"ENDPOINT_NAME":endpoint_name}
            }
        )
        self.function_arn = LambdaArn(response["FunctionArn"])
        self.wait_until_function_is_active()                
        print("You have successfully created your lambda function. Lambda Function arn:", response['FunctionArn'])
        os.remove("lambda_zipped.zip")
        return self.function_arn
    
    def use(self, data:dict):
        response = self.lambda_client.invoke(
        FunctionName=self.function_arn.resource,
        InvocationType='RequestResponse',  # Can be 'Event' for asynchronous invocation
        Payload=json.dumps(data).encode('utf-8')
        )
        output_dict = json.loads(response['Payload'].read().decode('utf-8'))
        return output_dict       
    
    def wait_until_function_is_active(self):
        if not self.function_arn:
            raise AttributeError("your SagemakerLambdaResourceUser does not have a function_arn yet. Make sure that you have deployed your lambda function first.")
        function_name = self.function_arn.resource
        while True:
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                status = response['Configuration']['State']
                
                if status == 'Active':
                    print(f"Lambda function {function_name} is now active.")
                    break
                elif status == 'Pending':
                    print(f"Lambda function {function_name} is still pending. Waiting...")
                    time.sleep(2)  # Wait for 5 seconds before checking again
                else:
                    print(f"Unexpected function state: {status}")
                    break
            except ClientError as e:
                print(f"An error occurred: {e}")
                break

if __name__ == "__main__":
    load_dotenv("../.env")
    role = RoleArn(os.environ["LAMBDA_ROLE_ARN"])
    lambdaDeployer = SageMakerLambdaResourceUser(role)
    lambdaDeployer.deploy("test", "test")
    #lambdaDeployer.wait_until_function_is_active()
    raw_response = lambdaDeployer.use({"text": "input in", "parameters":None})
    print(raw_response)
