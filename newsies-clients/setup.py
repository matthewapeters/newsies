"""
Setup for newsies-clients package
Database clients and session management
"""

from setuptools import setup, find_packages

setup(
    name="newsies-clients",
    version="0.2.0",
    description="Database clients and session management for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "chromadb>=0.4.0",
        "redis>=5.0.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
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
