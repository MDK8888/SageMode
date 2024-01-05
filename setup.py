from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='sagemode',
    version='0.1.0',
    author="MDK8888",
    description="Deploy, scale, and monitor your ML models all with one click. Native to AWS.",
    packages=find_packages(where="SageMode"),
    install_requires=parse_requirements('requirements.txt'),
    license="Apache License 2.0",
    package_data={
        '':["LICENSE", "requirements.txt"]
    }
)