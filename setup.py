import sys
import subprocess
from setuptools import setup, find_packages

subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])

import yaml

with open("requirements.yml") as file_handle:
    environment_data = yaml.full_load(file_handle)

setup(
    name="RuPersonaAgent",
    version="1.0.0",
    install_requires=environment_data['dependencies'][23]["pip"],
    packages=find_packages(include=['generative_model', 'speech_extraction',
                                    'inference_optimization', 'internet_memory_model',
                                    'knowledge_distillation', 'personification',
                                    'rule_based_information_extraction',
                                    'tests', 'datasets'])

)
