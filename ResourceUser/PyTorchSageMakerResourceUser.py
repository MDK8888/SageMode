import os
import time
import tarfile
from typing import Callable
import sagemaker
from sagemaker.s3 import S3Uploader
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorchModel
from Types.HFModels import model_types
from .ResourceUser import ResourceUser
from .LambdaResourceUser.SageMakerLambdaResourceUser import SageMakerLambdaResourceUser 
from Types.Arn import *
from Helpers.FileCopy import *
from Helpers.WriteFunctionToFile import write_function_to_file

class PyTorchSageMakerResourceUser(ResourceUser):

    def __init__(self, 
                 instance_type:str, 
                 next:dict[str, type] = None, 
                 previous:dict[str, type] = None, 
                 lambda_arn:LambdaArn = None):
        load_dotenv()
        role_arn = RoleArn(os.environ["SAGEMAKER_ROLE_ARN"])
        super().__init__(role_arn, next, previous)
        self.instance_type = instance_type
        self.lambda_user = SageMakerLambdaResourceUser(lambda_arn)  
    
    def create_bucket(self) -> None:
        print("Creating bucket...")
        sess = sagemaker.Session(self.boto3_session)
        try:
            self.bucket = sess.default_bucket()
        except:
            raise Exception("Unable to create bucket for session. Double check to make sure that your session is not 'None.'")

    def make_inference_local_directory(self, functions_dict:dict[str, Callable], model_path:str, weight_path:str, requirements_path:str = "requirements.txt") -> None:
        local_pytorch_directory_path = "PyTorchSageMaker"
        self.model_dir = local_pytorch_directory_path
        os.mkdir(local_pytorch_directory_path)
        for file_name in functions_dict:
            local_file_name = f"{file_name}.py"
            fn = functions_dict[file_name]
            write_function_to_file(fn, local_file_name)
            copy_file_to_directory(local_file_name, local_pytorch_directory_path, local_file_name)
        
        entry_file_path = f"{os.getcwd()}/InferenceFiles/PyTorchSageMaker.py"
        sagemaker_entry_point = "entry.py"
        copy_file_to_directory(entry_file_path, local_pytorch_directory_path, sagemaker_entry_point)

        sagemaker_model_path = "model.py"
        copy_file_to_directory(model_path, local_pytorch_directory_path, sagemaker_model_path)
        
        sagemaker_weight_path = "weights.pth"
        copy_file_to_directory(weight_path, local_pytorch_directory_path, sagemaker_weight_path)

        copy_file_to_directory(requirements_path, local_pytorch_directory_path, requirements_path)
        
        print(f"all necessary files copied into directory {local_pytorch_directory_path}/")

    def compress(self, output_file="model.tar.gz", skip=False) -> None:
        self.output_file = output_file
        if skip:
            print("You have selected to skip compressing your model. Skipping this step...")
        else:
            print("compressing directory...")
            t_start = time.time()
            parent_dir=os.getcwd()
            os.chdir(self.model_dir)
            with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
                for item in os.listdir('.'):
                    tar.add(item, arcname=item)
            print(f"compression finished successfully. Time taken: {time.time() - t_start:.2f} seconds")
            os.chdir(parent_dir)

    def upload_to_s3(self, skip=False) -> None:
        if skip:
            print("You have chosen to skip uploading to s3. Skipping this step...")
            self.model_uri = f"s3://{self.bucket}/{self.model_dir}/{self.output_file}"
        else:
            t_start = time.time()
            compressed_model_path = self.output_file
            self.model_uri = S3Uploader.upload(local_path=compressed_model_path, desired_s3_uri=f"s3://{self.bucket}/{self.model_dir}")
            print(f"upload to s3 finished successfully. Time taken: {time.time() - t_start:.2f} seconds")

    def deploy(self,function_name:str,
                    functions_dict:dict[str, Callable],
                    model_path:str = "model.py",
                    weight_path:str = "weights.pth",
                    skip_compression=False, 
                    skip_upload=False, 
                    python_version:str= "3.8",
                    pytorch_version:str= "1.10",
                    requirements_path:str = "requirements.txt",
                    timeout:int=3, 
                    ) -> LambdaArn:
        if self.lambda_user.function_arn:
            raise ValueError("We cannot call 'deploy' if the lambda_user already has a function_arn - set 'self.lambda_user.function_arn = None' and try again.")
        
        t_start = time.time()

        self.create_bucket()
        self.make_inference_local_directory(functions_dict, model_path, weight_path, requirements_path)
        self.compress("model.tar.gz", skip_compression)
        self.upload_to_s3(skip_upload)

        entry_file_in_directory = "entry.py"
        local_pytorch_directory_path = "PyTorchSageMaker"

        pytorch_model = PyTorchModel(
            model_data=self.model_uri,
            role=self.role_arn.raw_str,
            framework_version=pytorch_version,
            py_version="py38",
            entry_point=f"{local_pytorch_directory_path}/{entry_file_in_directory}",
            dependencies=[requirements_path]
        )

        self.predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type
        )
        
        function_arn:LambdaArn = self.lambda_user.deploy(function_name, self.predictor.endpoint_name, python_version, timeout)
        print(f"Deployment to SageMaker finished successfully. Time taken: {time.time() - t_start:.2f} seconds")
        return function_arn
    
    def use(self, data:dict):
        if not self.lambda_user.function_arn:
            raise AttributeError("You did not deploy a huggingface model as a lambda function on AWS. Please run .deploy() and try again.")
        self.check_input(data)
        response = self.lambda_user.use(data)
        self.check_output(response)
        return response