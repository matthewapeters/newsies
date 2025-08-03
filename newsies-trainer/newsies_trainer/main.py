"""
Main entry point for newsies-trainer service
"""

import sys
import uuid
import os
import pwd
from .pipelines import train_model_pipeline


def main():
    """Main entry point for trainer service"""
    print("Newsies Trainer Service")
    
    if len(sys.argv) < 2:
        print("Usage: python -m newsies_trainer.main <command>")
        print("Commands:")
        print("  train-model - Run the model training pipeline")
        return
    
    command = sys.argv[1]
    task_id = str(uuid.uuid4())
    user_id = pwd.getpwuid(os.getuid())[0]
    
    if command == "train-model":
        print(f"Starting train-model pipeline (task_id: {task_id})")
        train_model_pipeline(task_id=task_id)
        print("Train-model pipeline completed")
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
