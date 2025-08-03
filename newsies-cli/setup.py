"""
Setup for newsies-cli package
Command-line interface for Newsies
"""

from setuptools import setup, find_packages

setup(
    name="newsies-cli",
    version="0.2.0",
    description="Command-line interface for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "newsies-clients>=0.2.0",
        "click>=8.1.0",
        "rich>=13.6.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "newsies=newsies_cli.main:main",
        ],
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
