from setuptools import setup, find_packages

with open("requirements.txt", rb) as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="RuPersonaAgent",
    version="1.0.0",
    install_requires=requirements,
)