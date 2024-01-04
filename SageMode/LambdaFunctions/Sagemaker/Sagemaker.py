import json
import os
import boto3

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
runtime = boto3.client("runtime.sagemaker")

def lambda_handler(event, context):
    payload = json.dumps(event, indent=2).encode('utf-8')
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="application/json", Body=payload)
    result = json.loads(response["Body"].read().decode())
    return result