from setuptools import setup, find_packages

setup(
    name='sagemode',
    version='0.1.0',
    author="MDK8888",
    description="Deploy, scale, and monitor your ML models all with one click. Native to AWS.",
    packages=find_packages(),
    install_requires=['accelerate==0.25.0', 'annotated-types==0.6.0', 'anyio==4.2.0', 'arnparse==0.0.2', 'attrs==23.1.0', 'bcrypt==4.1.2', 'bitsandbytes==0.37.0', 'boto3==1.28.77', 'botocore==1.31.77', 'certifi==2023.7.22', 'cffi==1.16.0', 'charset-normalizer==3.3.2', 'click==8.1.7', 'cloudpickle==2.2.1', 'colorama==0.4.6', 'contextlib2==21.6.0', 'cryptography==41.0.7', 'diffusers==0.24.0', 'dill==0.3.7', 'dnspython==2.4.2', 'email-validator==2.1.0.post1', 'exceptiongroup==1.2.0', 'fastapi==0.108.0', 'filelock==3.13.1', 'fsspec==2023.10.0', 'google-pasta==0.2.0', 'h11==0.14.0', 'hf_transfer==0.1.3', 'httpcore==1.0.2', 'httptools==0.6.1', 'httpx==0.26.0', 'huggingface-hub==0.20.1', 'idna==3.4', 'importlib-metadata==6.8.0', 'itsdangerous==2.1.2', 'Jinja2==3.1.2', 'jmespath==1.0.1', 'jsonschema==4.19.2', 'jsonschema-specifications==2023.7.1', 'MarkupSafe==2.1.3', 'mpmath==1.3.0', 'multiprocess==0.70.15', 'networkx==3.2.1', 'numpy==1.26.1', 'orjson==3.9.10', 'packaging==23.2', 'pandas==2.1.2', 'paramiko==3.4.0', 'pathos==0.3.1', 'Pillow==10.1.0', 'platformdirs==3.11.0', 'pox==0.3.3', 'ppft==1.7.6.7', 'protobuf==4.25.0', 'psutil==5.9.6', 'pycparser==2.21', 'pydantic==2.5.2', 'pydantic-extra-types==2.2.0', 'pydantic-settings==2.1.0', 'pydantic_core==2.14.5', 'PyNaCl==1.5.0', 'python-dateutil==2.8.2', 'python-dotenv==1.0.0', 'python-multipart==0.0.6', 'pytz==2023.3.post1', 'PyYAML==6.0.1', 'referencing==0.30.2', 'regex==2023.10.3', 'requests==2.31.0', 'rpds-py==0.10.6', 's3transfer==0.7.0', 'safetensors==0.4.1', 'sagemaker==2.196.0', 'schema==0.7.5', 'six==1.16.0', 'smdebug-rulesconfig==1.0.1', 'sniffio==1.3.0', 'starlette==0.32.0.post1', 'sympy==1.12', 'tblib==1.7.0', 'tokenizers==0.13.3', 'torch==2.1.2', 'torchvision==0.16.2', 'tqdm==4.66.1', 'transformers==4.26.0', 'typing_extensions==4.8.0', 'tzdata==2023.3', 'ujson==5.9.0', 'urllib3==2.0.7', 'uvicorn==0.25.0', 'watchfiles==0.21.0', 'websockets==12.0', 'zipp==3.17.0'],
    license="Apache License 2.0",
    package_data={
        '':["LICENSE", "requirements.txt"]
    }
)