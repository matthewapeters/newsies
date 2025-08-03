"""
newsies-trainer
Model training and fine-tuning service
"""

__version__ = "0.2.0"

# Re-export commonly used items
# Note: Import only what exists in this package
try:
    from .pipelines.train_model import train_model_pipeline
except ImportError:
    train_model_pipeline = None

try:
    from .llm import *
except ImportError:
    pass

__all__ = [
    "pipelines",
    "llm",
    "train_model_pipeline",
]
