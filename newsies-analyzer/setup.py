"""
Setup for newsies-analyzer package
Content analysis and summarization service
"""

from setuptools import setup, find_packages

setup(
    name="newsies-analyzer",
    version="0.2.0",
    description="Content analysis service for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "newsies-clients>=0.2.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "spacy>=3.7.0",
        "nltk>=3.8.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.2.0",
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
