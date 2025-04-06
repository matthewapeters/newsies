"""
newsies.pipelines.train_model
"""

import os

from newsies.ap_news.archive import Archive, get_archive

from .task_status import TASK_STATUS


def train_model_pipeline():
    """
    train_model_pipeline
    """
    print("\nTRAIN MODEL\n")
    TASK_STATUS["train_model"] = "started"
    try:
        print("\n\t- training model\n")
        TASK_STATUS["train_model"] = "running - step: training model"
        archive: Archive = get_archive()
        # Load the latest training data
        batch = archive.get_latest_batch()
        # load_training_data()
        # Get the latest LoRA adapter
        # get_latest_lora_adapter()
        # Format the dataset for training
        # format_dataset()
        # Get the latest training data
        # get_latest_training_data()
        # Generate QA pairs for training
        # generate_qa_pairs()
        # Train the model
        # test_lora()
        TASK_STATUS["train_model"] = "complete"
    except Exception as e:
        TASK_STATUS["train_model"] = f"error: {e}"
        print(f"Error: {e}")
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
        # Add any additional cleanup or finalization steps as needed
        # For example, you might want to save the trained model or log the results
        # save_model()
        # log_results()
        # print("Model saved and results logged.")
        # This is a placeholder for any additional steps you might want to include


if __name__ == "__main__":
    # Run the training pipeline
    train_model_pipeline()
