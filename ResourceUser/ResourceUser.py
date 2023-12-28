import os
import boto3
from dotenv import load_dotenv
from Types.Arn import *
from abc import ABC, abstractmethod
from Helpers.TypeChecking import *

class ResourceUser(ABC):

    def __init__(self, role_arn:RoleArn, previous:dict[str, type]=None, next:dict[str, type]=None):
        load_dotenv()
        self.role_arn = role_arn
        self.access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        self.secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        self.boto3_session = boto3.Session(self.access_key_id, self.secret_access_key)
        if previous != None:
            self.previous = previous
        if next != None:
            self.next = next
        self.login()

    def __set_boto3_credentials(self, credentials:dict):
        try:
            os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
            os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
            self.boto3_session = boto3.Session(os.environ["AWS_ACCESS_KEY_ID"], 
                                               os.environ["AWS_SECRET_ACCESS_KEY"], 
                                               os.environ["AWS_SESSION_TOKEN"])
        except:
            raise ValueError("The dictionary that you passed in to '__set_boto3_credentials' does not contain the correct keys. \
                             Double check your input dictionary.")
        
    def login(self):
        client = boto3.client("sts")
        response = client.assume_role(
            RoleArn=self.role_arn.raw_str,
            RoleSessionName=f'{self.role_arn.resource}-session',
            DurationSeconds=3600
        )
        credentials = response["Credentials"]
        self.__set_boto3_credentials(credentials)
    
    def check_input(self, input_data:dict):
        if not hasattr(self, "previous"):
            return
        if not match_dicts(self.previous, input_data):
            raise ValueError("The data that you have passed to the use() function does not match the format specified when you initialized this resource user. Please check your \
                             input again to make sure that it matches.")
    
    def check_output(self, output_data:dict):
        if not hasattr(self, "next"):
            return
        if not match_dicts(self.next, output_data):
            raise ValueError("The response that your lambda function generated does not match the format specified when you initialized this resource user. Please check your \
                             output again to make sure that it matches.")

    @abstractmethod
    def deploy(self) -> LambdaArn:
        pass
    
    @abstractmethod
    def use(self):
        pass
