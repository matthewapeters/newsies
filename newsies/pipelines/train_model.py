"""
newsies.pipelines.train_model
"""

from uuid import uuid4
import os

# from typing import Dict, Tuple
from newsies.ap_news.archive import Archive, get_archive

from newsies.llm import (
    BatchSet,
    BatchRetriever,
    DataFramer,
    QuestionGenerator,
    DatasetFormatter,
    ModelTrainer,
)


from .task_status import TASK_STATUS

# pylint: disable=broad-exception-caught


def train_model_pipeline(task_id: str):
    """
    train_model_pipeline
    """
    print("\nTRAIN MODEL\n")
    TASK_STATUS[task_id] = "started"
    try:
        print("\n\t- training model\n")

        TASK_STATUS[task_id] = "running - step: building batches"
        archive: Archive = get_archive()
        batch_set: BatchSet = BatchSet(archive.build_batches())

        # Load the latest training data
        TASK_STATUS[task_id] = "running - step: retrieving metadata and embeddings"
        v = BatchRetriever()
        v.visit(batch_set)

        # convert batches to dataframes
        TASK_STATUS[task_id] = "running - step: formatting Data Sets"
        v = DataFramer()
        v.visit(batch_set)

        # generate training questions
        TASK_STATUS[task_id] = "running - step: generating training questions"
        v = QuestionGenerator(number_of_questions=5)
        v.visit(batch_set)

        # Split each publish date into train, test, token_train, and token_test
        # datasets.  These are saved to disk as parquet files.
        # The files are named with the publish date and the type of dataset.
        TASK_STATUS[task_id] = "running - step: format datasets"
        v = DatasetFormatter()
        v.visit(batch_set)

        # Train the model
        TASK_STATUS[task_id] = "running - step: training model"
        v = ModelTrainer()
        # setting these allows status update as we iterate through the batches
        # if either is not set, the status will not update
        v.status = TASK_STATUS
        v.task_id = task_id
        v.visit(batch_set)

        # test_lora()
        TASK_STATUS[task_id] = "complete"
    except Exception as e:
        TASK_STATUS[task_id] = f"error: {e}"
        print(f"Error: {e}")
        raise e
    finally:
        # Clean up any temporary files or directories created during the training process
        temp_dir = "./train_test"
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
        print("Temporary files cleaned up.")
        print("Training pipeline completed.")


def main():
    """main"""
    task_id = str(uuid4())
    train_model_pipeline(task_id=task_id)


if __name__ == "__main__":
    # Run the training pipeline
    main()
