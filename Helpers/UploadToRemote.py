import os

def upload_directory(local_directory, remote_directory, sftp):
    try:
        sftp.mkdir(remote_directory)
    except:
        print("Couldn't make remote directory.")
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            print(f"uploading file: {file}")
            local_file_path = os.path.join(root, file)
            remote_file_path = os.path.join(f"{remote_directory}/{file}")
            sftp.put(local_file_path, remote_file_path)
