from setuptools import find_packages, setup

setup(
    name="brain",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "tinydb",
        "pydantic",
        "cloudpickle",
        "requests",
        "rich",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "brain=cli:cli",
        ],
    },
    author="santosh kumar radha",
    author_email="instrument.santosh@gmail.com",
    description="A package for registering and executing AI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
