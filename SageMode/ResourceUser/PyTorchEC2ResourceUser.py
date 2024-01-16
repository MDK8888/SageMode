import os
import time
from dotenv import load_dotenv
from typing import Callable
import torch
import paramiko
from sagemode.Types.Arn import *
from sagemode.ResourceUser.ResourceUser import ResourceUser
from sagemode.ResourceUser.LambdaResourceUser.EC2LambdaResourceUser import EC2LambdaResourceUser
from sagemode.Helpers.WriteFunctionToFile import write_function_to_file
from sagemode.Helpers import copy_file_to_directory
from sagemode.Helpers import upload_directory
from sagemode.Helpers import wait_for_ssh_connection

class PyTorchEC2ResourceUser(ResourceUser):

    def __init__(self, lambda_function_arn:LambdaArn = None):

        load_dotenv(override=True)
        role_arn = RoleArn(os.environ["EC2_ROLE_ARN"])
        super().__init__(role_arn)
        self.ec2_client = self.boto3_session.client("ec2", region_name=os.environ["AWS_REGION"])
        self.lambda_user = EC2LambdaResourceUser(lambda_function_arn)

    def create_local_ec2_directory(self, model_path:str, 
                                        weight_path:str, 
                                        pre_process:Callable[[dict], torch.Tensor], 
                                        post_process:Callable[[torch.Tensor], dict], 
                                        requirements_path:str = "requirements.txt", 
                                        skip=False) -> None:
        
        if skip:
            print("You have selected to skip creating the local directory. Skipping this step...")
            return

        ec2_inference_path = os.path.join(os.getcwd(), "EC2InferenceLocal")
        os.mkdir(ec2_inference_path)

        server_code_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "InferenceFiles", "PyTorchEC2", "server", "main.py")
        ec2_server_file_name = "main.py"
        copy_file_to_directory(server_code_path, ec2_inference_path, ec2_server_file_name)

        pre_process_input_path = "pre_process.py"
        absolute_pre_process_input_path = f"{os.getcwd()}/pre_process.py"
        write_function_to_file(pre_process, absolute_pre_process_input_path)
        copy_file_to_directory(absolute_pre_process_input_path, ec2_inference_path, pre_process_input_path)
        
        post_process_output_path = "post_process.py"
        absolute_post_process_output_path = f"{os.getcwd()}/post_process.py"
        write_function_to_file(post_process, absolute_post_process_output_path)
        copy_file_to_directory(absolute_post_process_output_path, ec2_inference_path, post_process_output_path)

        ec2_weight_path = "weights.pth"
        absolute_weight_path = f"{os.getcwd()}/{weight_path}"
        copy_file_to_directory(absolute_weight_path, ec2_inference_path, ec2_weight_path)

        ec2_model_path = "model.py"
        absolute_model_path = f"{os.getcwd()}/{model_path}"
        copy_file_to_directory(absolute_model_path, ec2_inference_path, ec2_model_path)

        ec2_requirements_path = "requirements.txt"
        absolute_requirements_path = f"{os.getcwd()}/{requirements_path}"
        copy_file_to_directory(absolute_requirements_path, ec2_inference_path, ec2_requirements_path)
        
    def create_container_and_get_dns(self, ami_id:str, instance_type:str, container_dns:str=None, skip=False) -> str:
        if skip:
            if container_dns == None:
                raise ValueError("If you are selecting to skip creating your container, you must specify the dns of an already existing container.")
            
            print("You have selected to skip creating your container. Skipping this step.")
            return container_dns

        instance_id = self.ec2_client.run_instances(
        ImageId=ami_id,  # Specify the AMI ID
        MinCount=1,
        MaxCount=1,
        InstanceType=instance_type,  # Specify the instance type
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

    def upload_directory_to_ec2(self, public_dns:str, skip=True):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if skip:
            print("You have selected to skip uploading your directory to your ec2 instance. Skipping this step...")
            return

        max_retries = 3
        timeout = 5

        wait_for_ssh_connection(ssh_client, "ec2-user", max_retries, timeout, public_dns)

        print("Connection between local machine and EC2 container established.")

        local_directory = f'{os.getcwd()}/EC2InferenceLocal'
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

        print("Connection between local machine and ec2 instance established. The console output will now reflect your EC2 instance.")

        commands = [
            'sudo yum update',
            'sudo yum install -y python3 python3-pip',
            'python3 --version',
            f'cd EC2Inference && pip install -r requirements.txt && (nohup uvicorn main:app --host 0.0.0.0 --port {port} > output.log 2>&1 & disown)',  # Start FastAPI server
            'cat output.log'
        ]

        for command in commands:
            stdin, stdout, stderr = ssh_client.exec_command(command)
            if len(stderr.read().decode()) > 0:
                print(stderr.read().decode())
            else:
                print(stdout.read().decode())

        # Close the SSH connection
        ssh_client.close()

    def deploy(self,ami_id:str,
                    instance_type:str,
                    skips:list[str],
                    ec2_server_config:dict,
                    functions_dict:dict[str, Callable],
                    lambda_function_name:str = "lambda",
                    lambda_python_pip_prefix:list[str] = ["pip"] 
                    ) -> LambdaArn:
                
        if self.lambda_user.function_arn:
            raise ValueError("We cannot call 'deploy' if the lambda_user already has a function_arn - set 'self.lambda_user.function_arn = None' and try again.")
        
        pre_process, post_process = functions_dict["pre_process"], functions_dict["post_process"]
        model_path, weight_path, ec2_requirements_path, container_dns = \
            ec2_server_config["model_path"], ec2_server_config["weight_path"], ec2_server_config["requirements_path"], ec2_server_config.get("container_dns", None)

        skip_create_directory = "create_directory" in skips
        skip_create_container = "create_container" in skips
        skip_upload_directory = "upload_directory" in skips

        self.create_local_ec2_directory(model_path, weight_path, pre_process, post_process, ec2_requirements_path, skip_create_directory)
        public_dns = self.create_container_and_get_dns(ami_id, instance_type, container_dns, skip_create_container)
        self.upload_directory_to_ec2(public_dns, skip_upload_directory)
        self.run_server(public_dns)

        print("Server started on EC2 instance. Creating Lambda function...")

        function_arn:LambdaArn = self.lambda_user.deploy(lambda_function_name, 
                                                         public_dns, 
                                                         8000, 
                                                         lambda_python_pip_prefix
                                                         )
        print("Lambda function created. Deployment to EC2 complete.")
        return function_arn
    
    def use(self, data:dict):
        if not self.lambda_user.function_arn:
            raise AttributeError("You did not deploy a PyTorch model as a lambda function on AWS. Please run .deploy() and try again.")
        self.check_input(data)
        response = self.lambda_user.use(data)
        self.check_output(response)
        return response

    def teardown(self):
        pass