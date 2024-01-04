from typing import Callable
import inspect

def write_function_to_file(callback_function:Callable, filename:str) -> None:
    try:
        # Open the file in write mode
        with open(filename, 'w') as file:
            # Get the source code of the callback function
            source_code = inspect.getsource(callback_function)
            
            # Write the source code to the file
            file.write(source_code)
            
        print(f"Function written to {filename} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")





