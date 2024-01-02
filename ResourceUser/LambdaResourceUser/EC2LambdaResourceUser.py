import os
import subprocess
import time
import zipfile
import json
from botocore.exceptions import ClientError
from ..ResourceUser import ResourceUser
from Types.Arn import *
from Helpers.FileCopy import copy_file_to_directory

class EC2LambdaResourceUser(ResourceUser):

    def __init__(self, function_arn:LambdaArn=None):
        role_arn = RoleArn(os.environ["LAMBDA_ROLE_ARN"])
        super().__init__(role_arn)
        self.function_arn = function_arn
        self.lambda_client = self.boto3_session.client("lambda")
    
    def zip_lambda_file(self, lambda_function_file_name:str, python_pip_prefix:list[str], requests_version:str="2.31.0"):
        local_zipped_lambda_dir = "lambda_zipped"
        os.mkdir(local_zipped_lambda_dir)

        copy_file_to_directory(lambda_function_file_name, local_zipped_lambda_dir, "lambda_function.py")
        requirement_installation_suffix = ["install", f"requests=={requests_version}", "-t", local_zipped_lambda_dir]
        full_requirement_installation_command = python_pip_prefix + requirement_installation_suffix
        
        subprocess.run(full_requirement_installation_command, check=True)

        with zipfile.ZipFile(f"{local_zipped_lambda_dir}.zip", 'w') as zipped:
            for root, dirs, files in os.walk(local_zipped_lambda_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, local_zipped_lambda_dir)
                    zipped.write(file_path, arcname)

        print("Successfully zipped all files necessary for Lambda function. Now deploying...")

    def deploy(self, function_name:str,  
                    dns_name:str,
                    port:int=8000,
                    python_pip_prefix:list[str]=["pip"],
                    requests_version:str="2.31.0",
                    python_version:str="3.9",
                    timeout:int=3) -> LambdaArn:
        if self.function_arn:
            raise ValueError("This object cannot call 'deploy' if it already has a function_arn - set 'self.function_arn = None' and try again.")
        
        self.zip_lambda_file("LambdaFunctions/EC2/EC2.py", python_pip_prefix, requests_version)

        local_zipped_lambda_dir = "lambda_zipped.zip"

        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime=f'python{python_version}',
            Role=self.role_arn.raw_str,  # Replace with your Lambda execution role ARN
            Handler='lambda_function.lambda_handler',  # Specify the module and function name
            Timeout=timeout,
            Code={
                'ZipFile': open(local_zipped_lambda_dir, "rb").read()  # Read the content of the file
            },
            Environment={
                'Variables': {"DNS_NAME": dns_name, "PORT": str(port)}
            }
        )
        self.function_arn = LambdaArn(response["FunctionArn"])
        self.wait_until_function_is_active(timeout)                
        print("You have successfully deployed your Lambda function.")
        os.remove(local_zipped_lambda_dir)
        return self.function_arn
    
    def use(self, data:dict):
        response = self.lambda_client.invoke(
        FunctionName=self.function_arn.resource,
        InvocationType='RequestResponse',  # Can be 'Event' for asynchronous invocation
        Payload=json.dumps(data).encode('utf-8')
        )
        output_dict = json.loads(response['Payload'].read().decode('utf-8'))
        return output_dict       
    
    def wait_until_function_is_active(self, timeout:int=3):
        if not self.function_arn:
            raise AttributeError("your EC2LambdaResourceUser does not have a function_arn yet. Make sure that you have deployed your lambda function first.")
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
                    time.sleep(timeout)  # Wait for 5 seconds before checking again
                else:
                    print(f"Unexpected function state: {status}")
                    break
            except ClientError as e:
                print(f"An error occurred: {e}")
                break