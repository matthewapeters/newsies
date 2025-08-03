"""
Setup for newsies-common package
Shared utilities, data structures, and visitor pattern implementation
"""

from setuptools import setup, find_packages

setup(
    name="newsies-common",
    version="0.2.0",
    description="Shared utilities and data structures for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
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
