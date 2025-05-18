"""
newsies.llm.dataset_formatter
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    BatchEncoding,
)
from datasets import Dataset
import pandas as pd


from newsies.llm import BatchSet
from newsies.llm.batch_set_visitor import BatchSetVisitor

# pylint: disable=broad-exception-raised, broad-exception-caught

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]

BASE_DIR = "./train_test"

BOOKS = "📚"
CLEAN = "🧹"
DISK = "💾"
FAIL = "❌"
INFO = "ℹ️"
NEWSIES = "📰"
OK = "✅"
PACKAGE = "📦"
SEARCH = "🔍"
TRAINING = "🧠"
WAIT = "⏳"
WARN = "⚠️"


class DatasetFormatter(BatchSetVisitor):
    """
    DatasetFormatter
    """

    model_version = "7B-v0.3"
    base_model_name = f"mistralai/Mistral-{model_version}"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def __init__(self):
        super().__init__(
            BatchSet, "./training_data/formatted_dates.pkl", "split train-/test-data"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    @staticmethod
    def download_mistral():
        """
        download_mistral
        this should be used only in installation
        """

        mistral_models_path = Path.home().joinpath(
            "mistral_models", DatasetFormatter.model_version
        )
        if mistral_models_path.exists():
            print(
                f"{PACKAGE}{INFO} mistral models already downloaded to {mistral_models_path}"
            )
            return
        mistral_models_path.mkdir(parents=True, exist_ok=True)
        #
        snapshot_download(
            repo_id=DatasetFormatter.base_model_name,
            allow_patterns=[
                "params.json",
                "consolidated.safetensors",
                "tokenizer.model.v3",
            ],
            local_dir=mistral_models_path,
        )
        print(f"{PACKAGE}{OK} downloaded mistral models to {mistral_models_path}")

    def visit_batch_set(self, batch_set: BatchSet):
        """visit_batch_set"""
        # Make sure the model is downloaded
        DatasetFormatter.download_mistral()
        pub_date: int

        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR, exist_ok=True)

        for pub_date in batch_set.batches.keys():
            # Split to train and test
            if pub_date in self.history:
                print(f"{BOOKS}{INFO} Already split {pub_date}")
                self.update_status(f"Already split {pub_date}")
                continue
            print(f"{BOOKS}{INFO} Splitting {pub_date}")
            self.update_status(f"Splitting {pub_date}")
            start = datetime.now()
            try:
                self.history[pub_date] = batch_set[pub_date]
                self.split_train_and_test_data(pub_date)
                end = datetime.now()
                elapsed = end - start
                print(f"{BOOKS}{OK} {pub_date} took {elapsed}")
                self.update_status(f"{pub_date} took {elapsed}")
            except Exception as e:
                end = datetime.now()
                elapsed = end - start
                print(f"{BOOKS}{FAIL} {pub_date} failed: {e}")
                self.update_status(f"{pub_date} failed after {elapsed}: {e}")

            self.dump_history()
            print(f"{BOOKS}{OK} DatasetFormatter complete")

    def format_dataset(self, qa_dataset: pd.DataFrame) -> Dict[str, Dataset]:
        """
        Ensure tokenizer has a padding token and tokenize dataset.
        Args:
            qa_dataset (pd.DataFrame): DataFrame containing the dataset to be tokenized.
        Returns:
            Dict[str, Dataset]: Dictionary containing the 'train', 'test',
            'tokenized train',  and 'tokenized test' datasets.
        """

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )  # Use EOS token for padding

        def tokenize_sample(sample) -> BatchEncoding:
            """Tokenizes input and output text."""
            question = (
                str(sample["train_question"])
                if sample["train_question"] is not None
                else ""
            )
            answer = str(sample["answer"]) if sample["answer"] is not None else ""

            # Tokenize both question and answer with consistent padding length
            tokenized = self.tokenizer(
                question,
                text_target=answer,  # Proper way to tokenize input + labels
                padding="max_length",  # Force consistent padding
                truncation=True,
                max_length=512,
            )

            return tokenized  # Already contains input_ids, attention_mask, and labels

        # Drop rows where 'question' is NaN or empty
        qa_dataset = qa_dataset.dropna(subset=["question"])
        qa_dataset = qa_dataset[
            qa_dataset["question"].str.strip() != ""
        ]  # Remove empty questions

        dataset = Dataset.from_pandas(qa_dataset)
        split_dataset = dataset.train_test_split(test_size=0.2)

        split_dataset[TTRAIN] = split_dataset[TRAIN].map(
            tokenize_sample,
            remove_columns=[
                "train_question",
                "question",
                "answer",
                "uri",
                "section",
                "headline",
                "prompt",
                "doc",
            ],
        )
        split_dataset[TTEST] = split_dataset[TEST].map(
            tokenize_sample,
            remove_columns=[
                "train_question",
                "question",
                "answer",
                "uri",
                "section",
                "headline",
                "prompt",
                "doc",
            ],
        )

        return split_dataset

    def split_train_and_test_data(self, pub_date: int) -> Dict[str, Dict[str, Dataset]]:
        """
        get_train_and_test_data
        """
        # Apply the function
        train_df = load_training_data(batchset_index=pub_date)
        split_dataset = self.format_dataset(train_df)

        datehourminute = datetime.now().strftime(r"%Y%m%d%H%M")
        basedir = f"./{BASE_DIR}/{pub_date}/{datehourminute}"

        for d in TRAIN_DATA_TYPES:
            path = f"{basedir}/{d}"
            os.makedirs(f"{path}", exist_ok=True)
            try:
                print(f"{BOOKS}{INFO} Saving {path}")
                split_dataset[d].to_parquet(f"{path}/data.parquet", batch_size=1000)
                print(f"{BOOKS}{OK} Saved {path}/data.parquet")
            except Exception as e:
                print(f"{BOOKS}{FAIL} Failed to save {path}: {e}")
                raise e


def load_qa_from_parquet(file_path: str = None, batchset_index: int = None):
    """load_qa_from_parquet"""
    if batchset_index is not None:
        file_path = f"./training_data/{batchset_index:04d}"
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")


def load_training_data(batchset_index: int) -> pd.DataFrame:
    """
    load_training_data
    """
    project_root = os.path.abspath(".")
    if isinstance(batchset_index, str):
        batchset_index = int(batchset_index)
    try:
        df = pd.read_parquet(project_root + f"/training_data/{batchset_index:04d}")
        return df
    except FileNotFoundError as e:
        print(
            f"{BOOKS}{FAIL} File not found: {project_root}/training_data/{batchset_index:04d}: {e}"
        )
        raise e
