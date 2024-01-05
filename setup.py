from setuptools import setup, find_packages
import string

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        requirements_string = f.read()
        requirements_string = ''.join(filter(lambda x: x in string.printable, requirements_string))
        requirements_list = requirements_string.split("\n")
        requirements_list = [element for index, element in enumerate(requirements_list) if index % 2 == 0]
        return requirements_list

setup(
    name='sagemode',
    version='0.1.0',
    author="MDK8888",
    description="Deploy, scale, and monitor your ML models all with one click. Native to AWS.",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    license="Apache License 2.0",
    package_data={
        '':["LICENSE", "requirements.txt"]
    }
)