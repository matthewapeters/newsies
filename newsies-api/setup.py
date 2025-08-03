"""
Setup for newsies-api package
FastAPI gateway service and web dashboard
"""

from setuptools import setup, find_packages

setup(
    name="newsies-api",
    version="0.2.0",
    description="FastAPI gateway service for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "newsies-clients>=0.2.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "dash>=2.14.0",
        "plotly>=5.17.0",
        "pandas>=2.1.0",
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
