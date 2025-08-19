#!/usr/bin/env python3
"""
Setup script for Sine-Gordon Simulation and ML Localization Prediction
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sine-gordon-ml",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="Machine learning approach to predict energy localization in the Sine-Gordon equation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/Sine-Gordon-Simulation-ML-Prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: [Your License]",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
