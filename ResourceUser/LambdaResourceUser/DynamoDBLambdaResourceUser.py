import os
import zipfile
from ..ResourceUser import ResourceUser
from ...Types.Arn import *

class DynamoDBLambdaResourceUser(ResourceUser):

    def __init__(self, role_arn_str:str):
        super.__init__(role_arn_str)
    
    def zip_lambda_file(self, file_name:str, target_name:str):
        self.target_name = target_name
        with zipfile.ZipFile(f"{target_name}.zip", 'w') as zipped:
            zipped.write(f"{file_name}.py")
            zipped.close()

    def deploy(self, target_name:str, function_name:str, table_arn:str=None) -> LambdaArn:
        self.login()
        self.zip_lambda_file("Lambda Functions/DynamoDBLambda.py", target_name)
        if table_arn is None:
            table_arn = os.environ["table_arn"]

        self.lambda_client = self.boto3_session.client("lambda")
        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.8',
            Role=self.role_arn.raw_str,  # Replace with your Lambda execution role ARN
            Handler='Functions/DynamoDBLambda.py.lambda_handler',  # Specify the module and function name
            Code={
                'ZipFile': open(f"{self.target_name}.zip", "rb").read()  # Read the content of the file
            },
            Environment={
                'Variables': {"TABLE_ARN":table_arn}
            }
        )
        os.environ["lambdaArn"] = response["FunctionArn"]                
        print("You have successfully created your lambda function. Lambda Function arn:", response['FunctionArn'])
        return response["FunctionArn"]