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

    def __init__(self, instance_type:str, previous:dict[str, type]=None, next:dict[str, type]=None):
        load_dotenv()
        role_arn = RoleArn(os.environ["EC2_ROLE_ARN"])
        super().__init__(role_arn, previous, next)
        self.instance_type = instance_type
        self.ec2_client = self.boto3_session.client("ec2", region_name=os.environ["AWS_REGION"])
        self.lambda_user = EC2LambdaResourceUser()

    def create_ec2_directory(self, model_path:str, 
                            weight_path:str, 
                            pre_process:Callable[[dict], torch.Tensor], 
                            post_process:Callable[[torch.Tensor], dict], 
                            requirements_path:str = "requirements.txt") -> None:

        ec2_inference_path = "EC2InferenceLocal"
        os.mkdir(ec2_inference_path)

        server_code_path = "./InferenceFiles/server/main.py"
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

        print(f"Public DNS created. Public DNS: {public_dns}")

        return public_dns

    def upload_directory_to_ec2(self, public_dns:str):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        max_retries = 3
        timeout = 5

        wait_for_ssh_connection(ssh_client, "ec2-user", max_retries, timeout, public_dns)

        print("Connection between local machine and ec2 container established.")

        local_directory = 'EC2InferenceLocal'
        remote_directory = 'EC2Inference'        

        sftp = ssh_client.open_sftp()

        upload_directory(local_directory, remote_directory, sftp)

        # Close the SFTP connection
        sftp.close()

        # Close the SSH connection
        ssh_client.close()

        print("directory upload completed.")

    def run_server(self, public_dns:str):
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
            'cd EC2Inference && pip install -r requirements.txt && (nohup uvicorn main:app --host 0.0.0.0 --port 8000 > output.log 2>&1 & disown)',  # Start FastAPI server
        ]

        for command in commands:
            stdin, stdout, stderr = ssh_client.exec_command(command)
            print("stderr:", stderr.read().decode())
            print("stdout:", stdout.read().decode())

        # Close the SSH connection
        ssh_client.close()

    def deploy(self, ami_id: str, 
                    model_path:str, 
                    weight_path:str, 
                    pre_process:Callable[[dict], torch.Tensor], 
                    post_process:Callable[[torch.Tensor], dict], 
                    lambda_function_name:str,
                    lambda_timeout:int = 5,
                    python_pip_prefix:list[str] = ["pip"],
                    requirements_path:str = "requirements.txt") -> LambdaArn:
        
        self.create_ec2_directory(model_path, weight_path, pre_process, post_process, requirements_path)
        public_dns = self.create_container_and_get_dns(ami_id)
        self.upload_directory_to_ec2(public_dns)
        self.run_server(public_dns)

        function_arn:LambdaArn = self.lambda_user.deploy(lambda_function_name, 
                                                         public_dns, 
                                                         8000, 
                                                         python_pip_prefix, 
                                                         requirements_path, 
                                                         lambda_timeout)
        print("deployment to ec2 complete.")
        return function_arn
    
    def use():
        pass










        

        




        
        '''
        app = FastAPI()

        @app.post("/predict")
        def predict(data:dict) -> dict:
            pre_process_result = pre_process(data)
            raw_output = model(pre_process_result)
            post_process_result = post_process(raw_output)
            return post_process_result
    
        pickled_app = pickle.dumps(app)

        instance_response = self.ec2_client.run_instances(
        ImageId='ami-xxxxxxxxxxxxxxxxx',  # Specify the AMI ID
        InstanceType=self.instance_type,
        MinCount=1,
        MaxCount=1,
        UserData='#!/bin/bash\n',  # You can provide a user data script if needed
        )

        instance_id = instance_response['Instances'][0]['InstanceId']

        # Step 3: Wait for the instance to be running
        try:
            self.ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
        except WaiterError as e:
            print(f"Error waiting for instance to be running: {e}")
            return

        # Step 5: Get the public IP address of the instance
        instance_info = self.ec2_client.describe_instances(InstanceIds=[instance_id])
        public_ip = instance_info['Reservations'][0]['Instances'][0]['PublicIpAddress']

        # Step 6: Copy the pickled FastAPI app to the instance using paramiko
        key = paramiko.RSAKey(filename=f'/path/to/{key_name}.pem')  # Provide the path to your key file
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(public_ip, username='ec2-user', pkey=key)

        with ssh_client.open_sftp() as sftp:
            with sftp.file('/path/to/pickle_file', 'wb') as pickle_file:
                pickle_file.write(pickled_app)

        # Upload the requirements file (if it exists)
        try:
            with sftp.file('/path/to/requirements.txt', 'rb') as req_file:
                sftp.put(req_file, 'requirements.txt')
        except FileNotFoundError:
            pass  # Ignore if no requirements file is provided

        # Step 7: SSH into the instance and set up the environment
        # Install external dependencies and run the FastAPI app
        commands = [
            'sudo apt-get update',
            'sudo apt-get install -y python3-pip',
            'pip3 install -r /path/to/requirements.txt' if 'requirements.txt' in locals() else '',  # Install dependencies if requirements file exists
            f'python3 -m venv myenv && source myenv/bin/activate',  # Create and activate a virtual environment
            f'pip install uvicorn',
            f'python -c "import pickle; from fastapi import FastAPI; app = pickle.loads(open(\'/path/to/pickle_file\', \'rb\').read()); app.run(host=\'0.0.0.0\', port=8000)" &'
        ]
        command = ' && '.join(commands)

        stdin, stdout, stderr = ssh_client.exec_command(command)

        # Optionally, you may want to wait for some time to ensure that the app has started
        time.sleep(10)

        # Step 8: Print the public IP and port where the FastAPI app is running
        print(f"FastAPI app is running at: http://{public_ip}:8000")

        # Close the SSH connection
        ssh_client.close()
        '''

