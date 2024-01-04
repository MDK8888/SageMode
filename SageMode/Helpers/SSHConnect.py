import os
import time

def wait_for_ssh_connection(client, username, max_retries, check_freq_sec, public_dns):
    tries = 0
    ssh_client = client
    private_key_path = os.environ["PRIVATE_KEY_PATH"]
    while tries < max_retries:
        try:
            ssh_client.connect(public_dns, username=username, key_filename=private_key_path)
            return
        except:
            print("retrying connection...")
            time.sleep(check_freq_sec)
            tries+=1
    raise ConnectionRefusedError("Couldn't ssh into your remote server.")