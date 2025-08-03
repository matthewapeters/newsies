"""
Setup for newsies-trainer package
Model training and fine-tuning service
"""

from setuptools import setup, find_packages

setup(
    name="newsies-trainer",
    version="0.2.0",
    description="Model training service for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "newsies-clients>=0.2.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    author="Newsies Team",
    author_email="newsies@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
