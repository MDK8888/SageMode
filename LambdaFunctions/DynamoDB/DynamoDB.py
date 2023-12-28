import boto3
from boto3.dynamodb.conditions import Key
import os

def lambda_handler(event, context):
    # ARN of your DynamoDB table
    table_arn = os.environ["TABLE_ARN"]

    # Extract query parameters from the event
    partition_key_value = event.get('queryStringParameters', {}).get('partitionKey')
    sort_key_value = event.get('queryStringParameters', {}).get('sortKey')

    # Create a DynamoDB resource
    dynamodb = boto3.resource('dynamodb')

    # Create a DynamoDB table resource
    table = dynamodb.Table(table_arn)

    try:
        # Build the KeyConditionExpression based on the provided query parameters
        key_condition_expression = Key('partition_key_name').eq(partition_key_value)
        if sort_key_value:
            key_condition_expression = key_condition_expression & Key('sort_key_name').eq(sort_key_value)

        # Use the query method to query the table
        response = table.query(
            KeyConditionExpression=key_condition_expression
        )

        # Process the query result
        print('Query Result:', response)
        # Your logic to handle the result

        return {
            'statusCode': 200,
            'body': response
        }
    except Exception as e:
        print('Error:', e)

        return {
            'statusCode': 500,
            'body': {'error': 'Internal Server Error'}
        }