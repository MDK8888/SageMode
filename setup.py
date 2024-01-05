from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='sagemode',
    version='0.1.0',
    packages=find_packages(where="SageMode"),
    install_requires=parse_requirements('requirements.txt')
)