"""
Setup for newsies-scraper package
News scraping and article ingestion service
"""

from setuptools import setup, find_packages

setup(
    name="newsies-scraper",
    version="0.2.0",
    description="News scraping service for Newsies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "newsies-common>=0.2.0",
        "newsies-clients>=0.2.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "spacy>=3.7.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
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
