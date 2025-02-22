from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f.readlines()]

import newsies

_version = newsies.__version__

setup(
    name="newsies",
    version=_version,
    description="A news aggregator",
    author="Matthew A Peters",
    author_email="matthew@datadelve.net",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
)
