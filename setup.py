import os
from setuptools import setup, find_packages


# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


requirements = [
    "stable-baselines3==1.5.0",
    "tabulate>=0.8.0",
    "tensorboard>=2.9.0",
    "networkx>=2.8.4",
]

setup(
    name="NFVdeep",
    description="Deep Reinforcement Learning for Online Orchestration of Service Function Chains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CN-UPB/NFVdeep",
    packages=find_packages(),
    python_requires="==3.8.*",
    install_requires=requirements,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
