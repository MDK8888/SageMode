import sys
import boto3
from arnparse import arnparse

class RoleArn:

    def __init__(self, role_arn_str:str):
        self.raw_str = role_arn_str
        parsed_arn = arnparse(role_arn_str)
        self.account_id = parsed_arn.account_id
        self.region = parsed_arn.region
        self.resource = parsed_arn.resource
        if parsed_arn.resource_type != "role":
            raise ValueError("The string that you passed to create a RoleArn is invalid.")
        self.resource_type = "role"
        self.service = parsed_arn.service

    def verify_service(self, service:str) -> None:
        sts_client = boto3.client('sts')
        try:
            response = sts_client.assume_role(
            RoleArn=self.raw_str,
            RoleSessionName=f"{self.resource}-verification",
            DurationSeconds=3600
            )
        except:
            raise ValueError("The role that this arn is bound to does not have 'iam' permissions.")

        credentials = response['Credentials']
        access_key_id = credentials['AccessKeyId']
        secret_access_key = credentials['SecretAccessKey']
        session_token = credentials['SessionToken']

        iam_client = boto3.client(
        'iam',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token
        )

        role_response = iam_client.list_attached_role_policies(RoleName=self.resource)
        for policy in role_response['AttachedPolicies']:
            if policy["PolicyArn"] == service:
                sys.exit()
        raise ValueError("Your role arn does not have the appropriate permissions to invoke the service specified.")

class LambdaArn:

    def __init__(self, lambda_arn_str:str):
        self.raw_str = lambda_arn_str
        parsed_arn = arnparse(lambda_arn_str)
        self.account_id = parsed_arn.account_id
        self.region = parsed_arn.region
        self.resource = parsed_arn.resource
        self.resource_type = parsed_arn.resource_type
        if parsed_arn.service != "lambda":
            raise ValueError("The string that you passed to create a LambdaArn is invalid.")
        self.service = "lambda"

