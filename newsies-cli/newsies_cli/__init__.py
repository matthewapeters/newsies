"""
newsies-cli
Command-line interface for Newsies
"""

__version__ = "0.2.0"

# Re-export commonly used items
from .main import main
from .cli import *

__all__ = [
    "main",
    "cli",
]
