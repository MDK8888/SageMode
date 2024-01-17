import os
import time
import tarfile
from typing import Callable
import sagemaker
from sagemaker.s3 import S3Uploader
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorchModel
from sagemode.ResourceUser.ResourceUser import ResourceUser
from sagemode.ResourceUser.LambdaResourceUser.SageMakerLambdaResourceUser import SageMakerLambdaResourceUser 
from sagemode.Types.Arn import *
from sagemode.Helpers.FileCopy import *
from sagemode.Helpers import write_function_to_file

class PyTorchSageMakerResourceUser(ResourceUser):

    def __init__(self, lambda_arn:LambdaArn = None):

        load_dotenv(override=True)
        role_arn = RoleArn(os.environ["SAGEMAKER_ROLE_ARN"])
        super().__init__(role_arn)
        self.lambda_user = SageMakerLambdaResourceUser(lambda_arn)  
    
    def create_bucket(self) -> None:
        print("Creating bucket...")
        sess = sagemaker.Session(self.boto3_session)
        try:
            self.bucket = sess.default_bucket()
        except:
            raise Exception("Unable to create bucket for session. Double check to make sure that your session is not 'None.'")

    def make_inference_local_directory(self, functions_dict:dict[str, Callable], model_path:str, weight_path:str, requirements_path:str = "requirements.txt", skip=False) -> None:
        local_pytorch_directory_path = os.path.join(os.getcwd(), "PyTorchSageMaker")
        self.model_dir = str(local_pytorch_directory_path)
        
        if skip:
            print("You have selected to skip compressing your directory. Skipping this step...")
            return
        
        os.mkdir(local_pytorch_directory_path)

        for file_name in functions_dict:
            pytorch_directory_file_name = f"{local_pytorch_directory_path}/{file_name}.py"
            fn = functions_dict[file_name]
            write_function_to_file(fn, pytorch_directory_file_name)
        
        entry_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'InferenceFiles', "PyTorchSageMaker", "PyTorchSageMaker.py")
        sagemaker_entry_point = "entry.py"
        copy_file_to_directory(entry_file_path, local_pytorch_directory_path, sagemaker_entry_point)

        sagemaker_model_path = "model.py"
        absolute_model_path = f"{os.getcwd()}/{model_path}"
        copy_file_to_directory(absolute_model_path, local_pytorch_directory_path, sagemaker_model_path)
        
        sagemaker_weight_path = "weights.pth"
        absolute_weight_path = f"{os.getcwd()}/{weight_path}"
        copy_file_to_directory(absolute_weight_path, local_pytorch_directory_path, sagemaker_weight_path)

        absolute_requirements_path = f"{os.getcwd()}/{requirements_path}"
        copy_file_to_directory(absolute_requirements_path, local_pytorch_directory_path, requirements_path)
        
        print(f"all necessary files copied into directory {local_pytorch_directory_path}/")

    def compress(self, output_file="model.tar.gz", skip=False) -> None:
        self.output_file = str(os.path.join(os.getcwd(), output_file))
        if skip:
            print("You have selected to skip compressing your model. Skipping this step...")
        else:
            print("compressing directory...")
            t_start = time.time()
            parent_dir=os.getcwd()
            os.chdir(self.model_dir)
            with tarfile.open(self.output_file, "w:gz") as tar:
                for item in os.listdir('.'):
                    tar.add(item, arcname=item)
            print(f"compression finished successfully. Time taken: {time.time() - t_start:.2f} seconds")
            os.chdir(parent_dir)

    def upload_to_s3(self, skip=False) -> None:
        s3_model_dir = os.path.basename(self.model_dir)
        s3_output_file = os.path.basename(self.output_file)

        if skip:
            print("You have chosen to skip uploading to s3. Skipping this step...")
            self.model_uri = f"s3://{self.bucket}/{s3_model_dir}/{s3_output_file}"
        else:
            t_start = time.time()
            compressed_model_path = self.output_file
            self.model_uri = S3Uploader.upload(local_path=compressed_model_path, desired_s3_uri=f"s3://{self.bucket}/{s3_model_dir}")
            print(f"upload to s3 finished successfully. Time taken: {time.time() - t_start:.2f} seconds")

    def deploy(self, model_path:str,
                     weight_path:str,
                     instance_type:str,
                     functions_dict: dict[str, Callable],
                     skips:list[str] = [],
                     sagemaker_env_config:dict[str, str] = {"python_version": "py38", "pytorch_version":"1.10", "requirements_path":"requirements.txt"}, 
                     function_name:str = "lambda",
                     lambda_env_config:dict[str, str] = {"python_version": "3.8", "timeout": 3}
                    ) -> LambdaArn:
        
        if self.lambda_user.function_arn:
            raise ValueError("We cannot call 'deploy' if the lambda_user already has a function_arn - set 'self.lambda_user.function_arn = None' and try again.")
        
        t_start = time.time()
    
        sagemaker_python_version, sagemaker_pytorch_version, sagemaker_requirements_path = \
            sagemaker_env_config["python_version"], sagemaker_env_config["pytorch_version"], sagemaker_env_config["requirements_path"]

        skip_make_directory = "make_directory" in skips
        skip_compression = "compression" in skips
        skip_upload = "upload" in skips

        self.create_bucket()
        self.make_inference_local_directory(functions_dict, model_path, weight_path, sagemaker_requirements_path, skip_make_directory)
        self.compress("model.tar.gz", skip_compression)
        self.upload_to_s3(skip_upload)

        entry_file_in_directory = "entry.py"
        local_pytorch_directory_path = "PyTorchSageMaker"

        pytorch_model = PyTorchModel(
            model_data=self.model_uri,
            role=self.role_arn.raw_str,
            framework_version=sagemaker_pytorch_version,
            py_version=sagemaker_python_version,
            entry_point=f"{os.getcwd()}/{local_pytorch_directory_path}/{entry_file_in_directory}",
            dependencies=[f"{os.getcwd()}/{sagemaker_requirements_path}"]
        )

        self.predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type
        )
        
        function_arn:LambdaArn = self.lambda_user.deploy(function_name, self.predictor.endpoint_name, lambda_env_config)
        print(f"Deployment to SageMaker finished successfully. Time taken: {time.time() - t_start:.2f} seconds")
        return function_arn
    
    def use(self, data:dict):
        if not self.lambda_user.function_arn:
            raise AttributeError("You did not deploy a huggingface model as a lambda function on AWS. Please run .deploy() and try again.")
        self.check_input(data)
        response = self.lambda_user.use(data)
        self.check_output(response)
        return response

    def teardown(self):
        lambda_environment_variables = self.lambda_user.get_env()
        self.lambda_user.teardown()
        sagemaker_endpoint_name = lambda_environment_variables["ENDPOINT_NAME"]

        sagemaker_client = self.boto3_session.client("sagemaker")
        sagemaker_client.delete_endpoint(EndpointName=sagemaker_endpoint_name)

        print("Your SageMaker endpoint has been deleted.")