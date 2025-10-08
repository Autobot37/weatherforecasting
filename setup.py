from setuptools import setup, find_packages

setup(
    name='weatherforecasting',
    version='0.1.0',
    description='A deep learning pipeline for weather forecasting research.',
    author='bagri',
    packages=find_packages(),
)

#!pip install -e .
print("""
> - run from root weatherforecasting all paths are realative to it.
> - package is pipeline all things which will be imported resides here.
> - exp_name is model + dataset + some hyperparams
> - each dataset should have a plot fn and data dict it should take and return a figure# weatherforecasting
> - explicit params are passed.
> - each train script should be flexible and helpers used for specialised functions even though many of them but write inputs and outputs.
""")