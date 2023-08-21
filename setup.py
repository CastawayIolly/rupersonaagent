import sys
import subprocess
from setuptools import setup, find_packages
import yaml

subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])

with open("requirements.yml") as file_handle:
    environment_data = yaml.full_load(file_handle)

setup(
    name="RuPersonaAgent",
    version="1.0.0",
    install_requires=environment_data['dependencies'][23]["pip"],
    packages=find_packages(include=['generative_model', 'speech_extraction'])

)
