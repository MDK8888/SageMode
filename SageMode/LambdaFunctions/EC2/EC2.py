import os
import json
import requests

DNS = os.environ["DNS_NAME"]
PORT = os.environ["PORT"]

def lambda_handler(event, context):
    payload = json.dumps(event, indent=2).encode('utf-8')
    payload = json.loads(payload)
    server_url = f"http://{DNS}:{PORT}/predict"
    response = requests.post(server_url, json=payload)
    json_response = response.json()
    return json_response
