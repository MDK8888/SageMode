from pathlib import Path
import os
import time
import tarfile
import sagemaker
from sagemaker.s3 import S3Uploader
from dotenv import load_dotenv
from shutil import rmtree
from distutils.dir_util import copy_tree
from huggingface_hub import snapshot_download
from sagemaker.huggingface.model import HuggingFaceModel
from Types.HFModels import model_types
from .ResourceUser import ResourceUser
from .LambdaResourceUser.SagemakerLambdaResourceUser import SageMakerLambdaResourceUser 
from Types.Arn import *
from Helpers.FileCopy import *

class HFSageMakerResourceUser(ResourceUser):

    def __init__(self, instance_type:str ,next:dict[str, type] = None, previous:dict[str, type] = None, lambda_arn:LambdaArn = None):
        load_dotenv()
        role_arn = RoleArn(os.environ["SAGEMAKER_ROLE_ARN"])
        super().__init__(role_arn, next, previous)
        self.instance_type = instance_type
        self.lambda_deployer = SageMakerLambdaResourceUser(lambda_arn)  
    
    def create_bucket(self) -> None:
        print("creating bucket...")
        sess = sagemaker.Session(self.boto3_session)
        try:
            self.bucket = sess.default_bucket()
        except:
            raise Exception("unable to create bucket for session. Double check to make sure that your session is not 'None.'")

    def copy_from_huggingface(self, model_id:str) -> None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        model_tar_dir = Path(model_id.split("/")[-1])
        model_tar_dir.mkdir(exist_ok=True)
        t_start = time.time()
        try:
            snapshot_download(model_id, local_dir=str(model_tar_dir), local_dir_use_symlinks=False)
            print(f"huggingface model copied successfully. Time taken: {time.time() - t_start:.2f} seconds")
        except:
            raise ValueError("the model_id you have specified does not exist.")

        for model_type in model_types:
            try:
                model_type.from_pretrained(model_id)
                inference_file_name = model_type.__name__
                copy_file_to_directory(f"InferenceFiles/{inference_file_name}.py", "code", "inference.py")    
                break
            except:
                continue

        self.model_dir = model_tar_dir
        # copy code/ to model dir
        copy_tree("code/", str(model_tar_dir.joinpath("code")))
        rmtree("code")

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

    def deploy(self, model_id:str, function_name:str, timeout:int=3, skip_compression=False, skip_upload=False, deployment_config:dict={"transformers_version":"4.26", 
                                                                                               "pytorch_version":"1.13", 
                                                                                               "python_version":"py39"}) -> LambdaArn:
        self.create_bucket()
        self.copy_from_huggingface(model_id)
        self.compress("model.tar.gz", skip_compression)
        self.upload_to_s3(skip_upload)


        transformers_version, pytorch_version, python_version = \
        deployment_config["transformers_version"], deployment_config["pytorch_version"], deployment_config["python_version"]

        huggingface_model = HuggingFaceModel(
        model_data=self.model_uri,      # path to your model and script
        role=self.role_arn.raw_str,   # iam role with permissions to create an Endpoint
        transformers_version=transformers_version,  # transformers version used
        pytorch_version=pytorch_version,       # pytorch version used
        py_version=python_version,            # python version used
        )

        t_start = time.time()
        predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=self.instance_type
        )
        function_arn:LambdaArn = self.lambda_deployer.deploy(function_name, predictor.endpoint_name, timeout)
        print(f"deployment to sagemaker finished successfully. Time taken: {time.time() - t_start:.2f} seconds")
        return function_arn
    
    def use(self, data:dict):
        if not self.lambda_deployer.function_arn:
            raise AttributeError("You did not deploy a huggingface model as a lambda function on AWS. Please run .deploy() and try again.")
        self.lambda_deployer.wait_until_function_is_active()
        self.check_input(data)
        response = self.lambda_deployer.use(data)
        self.check_output(response)
        return response