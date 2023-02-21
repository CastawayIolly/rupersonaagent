
from setuptools import setup, find_packages

import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'PyYAML'])
import yaml

with open("requirements.yml") as file_handle:
    environment_data = yaml.full_load(file_handle)

setup(
    name="RuPersonaAgent",
    version="1.0.0",
    install_requires=environment_data['dependencies'][23]["pip"],
    packages=find_packages(include=['generative_model', 'speech_extraction', 'rule_based_information_extraction'])

)