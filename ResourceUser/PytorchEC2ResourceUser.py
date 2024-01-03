import os
import time
from dotenv import load_dotenv
from typing import Callable
import torch
import paramiko
from Types.Arn import *
from .ResourceUser import ResourceUser
from ResourceUser.LambdaResourceUser.EC2LambdaResourceUser import EC2LambdaResourceUser
from Helpers.WriteFunctionToFile import write_function_to_file
from Helpers.FileCopy import copy_file_to_directory
from Helpers.UploadToRemote import upload_directory
from Helpers.SSHConnect import wait_for_ssh_connection

class PyTorchEC2ResourceUser(ResourceUser):

    def __init__(self, instance_type:str, previous:dict[str, type] = None, next:dict[str, type] = None, lambda_function_arn:LambdaArn = None):
        load_dotenv()
        role_arn = RoleArn(os.environ["EC2_ROLE_ARN"])
        super().__init__(role_arn, previous, next)
        self.instance_type = instance_type
        self.ec2_client = self.boto3_session.client("ec2", region_name=os.environ["AWS_REGION"])
        self.lambda_user = EC2LambdaResourceUser(lambda_function_arn)

    def create_local_ec2_directory(self, model_path:str, 
                            weight_path:str, 
                            pre_process:Callable[[dict], torch.Tensor], 
                            post_process:Callable[[torch.Tensor], dict], 
                            requirements_path:str = "requirements.txt") -> None:

        ec2_inference_path = "EC2InferenceLocal"
        os.mkdir(ec2_inference_path)

        server_code_path = f"{os.getcwd()}/InferenceFiles/PyTorchEC2/server/main.py"
        ec2_server_file_name = server_code_path.split("/")[-1]
        copy_file_to_directory(server_code_path, ec2_inference_path, ec2_server_file_name)

        pre_process_input_path = "pre_process.py"
        write_function_to_file(pre_process, pre_process_input_path)
        copy_file_to_directory(pre_process_input_path, ec2_inference_path, pre_process_input_path)
        
        post_process_output_path = "post_process.py"
        write_function_to_file(post_process, post_process_output_path)
        copy_file_to_directory(post_process_output_path, ec2_inference_path, post_process_output_path)

        ec2_weight_path = "weights.pth"
        copy_file_to_directory(weight_path, ec2_inference_path, ec2_weight_path)

        ec2_model_path = "model.py"
        copy_file_to_directory(model_path, ec2_inference_path, ec2_model_path)

        ec2_requirements_path = "requirements.txt"
        copy_file_to_directory(requirements_path, ec2_inference_path, ec2_requirements_path)
        
    def create_container_and_get_dns(self, ami_id:str) -> str:
        instance_id = self.ec2_client.run_instances(
        ImageId=ami_id,  # Specify the AMI ID
        MinCount=1,
        MaxCount=1,
        InstanceType=self.instance_type,  # Specify the instance type
        KeyName=os.environ["EC2_KEY_PAIR_NAME"],  # Specify your key pair
        SecurityGroupIds=[os.environ["SECURITY_GROUP_ID"]]
        )["Instances"][0]["InstanceId"]

        ec2_resource = self.boto3_session.resource("ec2")
        instance = ec2_resource.Instance(instance_id)

        instance.wait_until_running()

        instance.reload()

        public_dns = instance.public_dns_name

        print(f"Public DNS created.")

        return public_dns

    def upload_directory_to_ec2(self, public_dns:str):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        max_retries = 3
        timeout = 5

        wait_for_ssh_connection(ssh_client, "ec2-user", max_retries, timeout, public_dns)

        print("Connection between local machine and EC2 container established.")

        local_directory = 'EC2InferenceLocal'
        remote_directory = 'EC2Inference'        

        sftp = ssh_client.open_sftp()

        upload_directory(local_directory, remote_directory, sftp)

        # Close the SFTP connection
        sftp.close()

        # Close the SSH connection
        ssh_client.close()

        print("Directory upload completed.")

    def run_server(self, public_dns:str, port:int=8000):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        max_retries = 3
        timeout = 5

        wait_for_ssh_connection(ssh_client, "ec2-user", max_retries, timeout, public_dns)

        print("Connection between local machine and ec2 instance established. Executing commands...")

        commands = [
            'sudo yum update',
            'sudo yum install -y python3 python3-pip',
            'python3 --version',
            f'cd EC2Inference && pip install -r requirements.txt && (nohup uvicorn main:app --host 0.0.0.0 --port {port} > output.log 2>&1 & disown)',  # Start FastAPI server
        ]

        for command in commands:
            stdin, stdout, stderr = ssh_client.exec_command(command)
            if len(stderr.read().decode()) > 0:
                print(stderr.read().decode())
            else:
                print(stdout.read().decode())

        # Close the SSH connection
        ssh_client.close()

    def deploy(self, ami_id: str, 
                    model_path:str, 
                    weight_path:str, 
                    pre_process:Callable[[dict], torch.Tensor], 
                    post_process:Callable[[torch.Tensor], dict], 
                    lambda_function_name:str,
                    lambda_python_pip_prefix:list[str] = ["pip"],
                    ec2_requirements_path:str = "requirements.txt", 
                    ) -> LambdaArn:
                
        if self.lambda_user.function_arn:
            raise ValueError("We cannot call 'deploy' if the lambda_user already has a function_arn - set 'self.lambda_user.function_arn = None' and try again.")
        
        self.create_local_ec2_directory(model_path, weight_path, pre_process, post_process, ec2_requirements_path)
        public_dns = self.create_container_and_get_dns(ami_id)
        self.upload_directory_to_ec2(public_dns)
        self.run_server(public_dns)

        print("Server started on EC2 instance. Creating Lambda function...")

        function_arn:LambdaArn = self.lambda_user.deploy(lambda_function_name, 
                                                         public_dns, 
                                                         8000, 
                                                         lambda_python_pip_prefix
                                                         )
        print("Lambda function created. Deployment to ec2 complete.")
        return function_arn
    
    def use(self, data:dict):
        if not self.lambda_user.function_arn:
            raise AttributeError("You did not deploy a PyTorch model as a lambda function on AWS. Please run .deploy() and try again.")
        self.check_input(data)
        response = self.lambda_user.use(data)
        self.check_output(response)
        return response