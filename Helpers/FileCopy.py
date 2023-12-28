import shutil
import os

def copy_file_to_directory(file_name, destination_directory, new_name):
    # Check if the file exists
    if not os.path.isfile(file_name):
        print(f"Error: The file '{file_name}' does not exist.")
        return

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Create the new file path in the destination directory with the specified name
    new_file_path = os.path.join(destination_directory, new_name)

    # Copy the contents of the file to the new location
    shutil.copy(file_name, new_file_path)

    print(f"File '{file_name}' copied to '{new_file_path}' successfully.")
