"""
newsies.llm.dataset_formatter
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict

from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    BatchEncoding,
)
from datasets import Dataset
import pandas as pd


from newsies.llm import BatchSet

# pylint: disable=broad-exception-raised, broad-exception-caught

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]


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
    df = pd.read_parquet(project_root + f"/training_data/{batchset_index:04d}")
    return df


class DatasetFormatter:
    """
    DatasetFormatter
    """

    model_version = "7B-v0.3"
    base_model_name = f"mistralai/Mistral-{model_version}"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    @staticmethod
    def download_mistral():
        """
        download_mistral
        this should be used only in installation
        """

        mistral_models_path = Path.home().joinpath(
            "mistral_models", DatasetFormatter.model_version
        )
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

    def visit(self, o: Any):
        """visit"""
        match o:
            case BatchSet():
                o.accept(self)
            case _:
                raise TypeError(f"BatchRetriever only accepts BatchSet, got {type(o)}")

    def visit_batch_set(self, batch_set: BatchSet):
        """visit_batch_set"""

    def format_dataset(self, qa_dataset: pd.DataFrame):
        """Ensure tokenizer has a padding token and tokenize dataset."""

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

    def get_train_and_test_data(
        self, batchset_id: int
    ) -> Dict[str, Dict[str, Dataset]]:
        """
        get_train_and_test_data
        """
        # Apply the function
        train_df = load_training_data(batchset_id)
        split_dataset = self.format_dataset(train_df)

        datehourminute = datetime.now().strftime(r"%Y%m%d%H%M")
        basedir = f"./train_test/{datehourminute}"

        for d in TRAIN_DATA_TYPES:
            os.makedirs(f"{basedir}/{d}", exist_ok=True)
            split_dataset[d].to_parquet(f"{basedir}/{d}/data.parquet", batch_size=1000)

        return split_dataset
