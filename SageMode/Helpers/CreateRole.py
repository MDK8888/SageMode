import os
import json
import boto3
from dotenv import load_dotenv

def create_role(role_name:str, services:list[str] = None, primary_service:str = None) -> str:
    load_dotenv(override=True)
    access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    boto3_session = boto3.Session(access_key_id, secret_access_key)

    if primary_service is None:
        primary_service = services[0]

    iam_client = boto3_session.client("iam")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "AWS": os.environ["USER_ARN"],
                    "Service": f"{primary_service.lower()}.amazonaws.com" 
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    service_policy_map = {
        "Lambda": "AWSLambda_",
        "SageMaker": "AmazonSageMaker",
        "EC2": "AmazonEC2",
        "S3": "AmazonS3",
        "Step_Functions": "AWSStepFunctions"
    }

    create_role_response = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy)
    )

    for service in services:
    
        if not service in service_policy_map:
            raise ValueError("You are trying to create a role for an unsupported service.")
    
        actual_aws_policy_prefix = service_policy_map[service]

        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn=f'arn:aws:iam::aws:policy/{actual_aws_policy_prefix}FullAccess'
        )

    role_arn = create_role_response['Role']['Arn']

    print("IAM Role created successfully. Role arn:", role_arn)
    return role_arn





