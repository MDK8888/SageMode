import os
from .ResourceUser import ResourceUser
from ..Types.Arn import *

class APIGatewayResourceUser(ResourceUser):
    
    def __init__(self, role_arn:RoleArn):
        super.__init__(role_arn)

    def deploy(self, server_name:str, path_part:str, stage_name:str, lambda_arn:str=None):
        self.login()
        if lambda_arn is None:
            lambda_arn = os.environ["lambdaArn"]

        apigateway_client = self.boto3_session.client('apigateway')

        # Create REST API
        api = apigateway_client.create_rest_api(
            name=server_name,
            description="Generic Description"
        )

        # Get root resource ID
        root_id = apigateway_client.get_resources(restApiId=api['id'])['items'][0]['id']

        # Create a resource
        resource = apigateway_client.create_resource(
            restApiId=api['id'],
            parentId=root_id,
            pathPart=path_part
        )

        lambda_uri = f"arn:aws:apigateway:{os.environ['AWS_REGION']}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"

        apigateway_client.put_method(
            restApiId=api['id'],
            resourceId=resource['id'],
            httpMethod='POST',
            authorizationType='NONE'
        )
        
        response = apigateway_client.put_integration(
            restApiId=api['id'],
            resourceId=resource['id'],
            httpMethod='POST',
            type='AWS',
            integrationHttpMethod='POST',
            uri=lambda_uri,
            credentials=self.role_arn
        )

        response = apigateway_client.put_integration_response(
            restApiId=api['id'],
            resourceId=resource['id'],
            httpMethod='POST',
            statusCode='200',
            responseTemplates={
                'application/json': ""  # Replace with your response template, if needed
            }
        )

        response = apigateway_client.put_method_response(
            restApiId=api['id'],
            resourceId=resource['id'],
            httpMethod='POST',
            statusCode='200',
            responseModels={
                'application/json': 'Empty'
            }
        )
        
        apigateway_client.create_deployment(restApiId=api['id'], stageName=stage_name)
        aws_region = os.environ["AWS_REGION"]
        invoke_url = f"https://{api['id']}.execute-api.{aws_region}.amazonaws.com/{stage_name}/{path_part}"
        print("AWS API Gateway Created. This is the url to invoke to call your model:", invoke_url)
        return invoke_url