import os
import time
from dotenv import load_dotenv
from typing import Callable
import torch
from torch import nn
import paramiko
from botocore.exceptions import WaiterError
from Types.Arn import *
from .ResourceUser import ResourceUser
from Helpers.WriteFunctionToFile import write_function_to_file
from Helpers.FileCopy import copy_file_to_directory

class PytorchEC2ResourceUser(ResourceUser):

    def __init__(self, instance_type:str, previous:dict[str, type]=None, next:dict[str, type]=None):
        load_dotenv()
        role_arn = RoleArn(os.environ["EC2_ROLE_ARN"])
        super().__init__(role_arn, previous, next)
        self.instance_type = instance_type
        self.ec2_client = self.boto3_session.client("ec2", region_name=os.environ["AWS_REGION"])

    def deploy(self, model_path:str, 
                    weight_path:str, 
                    pre_process:Callable[[dict], torch.Tensor], 
                    post_process:Callable[[torch.Tensor], dict], 
                    requirements_path:str = "requirements.txt") -> LambdaArn:

        def create_ec2_directory(model_path:str, 
                                weight_path:str, 
                                pre_process:Callable[[dict], torch.Tensor], 
                                post_process:Callable[[torch.Tensor], dict], 
                                requirements_path:str = "requirements.txt") -> None:

            ec2_inference_path = "EC2Inference"
            server_code_path = "./InferenceFiles/server/main.py"
            ec2_server_file_name = server_code_path.split("/")[-1]

            os.mkdir(ec2_inference_path)
            copy_file_to_directory(server_code_path, ec2_inference_path, ec2_server_file_name)

            pre_process_input_path = "pre_process.py"
            write_function_to_file(pre_process, pre_process_input_path)
            copy_file_to_directory(pre_process_input_path, ec2_inference_path, pre_process_input_path)
            
            post_process_output_path = "post_process.py"
            write_function_to_file(post_process, post_process_output_path)
            copy_file_to_directory(post_process_output_path, ec2_inference_path, post_process_output_path)

            ec2_weight_path = "weights.pth"
            os.rename(weight_path, ec2_weight_path)
            copy_file_to_directory(ec2_weight_path, ec2_inference_path, ec2_weight_path)

            ec2_model_path = "model.py"
            os.rename(model_path, ec2_model_path)
            copy_file_to_directory(ec2_model_path, ec2_inference_path, ec2_model_path)

            ec2_requirements_path = "requirements.txt"
            os.rename(requirements_path, ec2_requirements_path)
            copy_file_to_directory(ec2_requirements_path, ec2_inference_path, ec2_requirements_path)
        
        create_ec2_directory(model_path, weight_path, pre_process, post_process, requirements_path)

        










        

        




        
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
    
    def use():
        pass
            

