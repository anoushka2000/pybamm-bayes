from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# base requirements
install_requires = open(path.join(here, "requirements.txt")).read().strip().split("\n")

setup(
    name='battery-model-parameterization',
    version='0.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Battery model parameter identifiability project.'
)
