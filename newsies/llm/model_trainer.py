"""
newsies.llm.model_trainer
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple
import os
import pickle

from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from newsies.llm.batch_set import BatchSet

# pylint: disable=dangerous-default-value, broad-exception-caught

TRAINED_DATES_PATH = "./train_test/trained_dates.pkl"
_BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
_TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]


def get_latest_training_data(
    pub_date: int, types: List[str] = _TRAIN_DATA_TYPES
) -> Dict[str, pd.DataFrame]:
    """
    get_latest_train_data
    """
    try:
        base_dir = f"./train_test/{pub_date:04d}"
        dirs = os.listdir(base_dir)
        if len(dirs) == 0:
            raise OSError(
                "No test data found. Please run get_train_and_test_data() first."
            )
        dirs.sort()
        latest_dir = dirs[-1]
        train_dict: Dict[str, Dataset] = {}
        for d in types:
            train_dict[d] = Dataset.from_pandas(
                pd.read_parquet(
                    f"{base_dir}/{latest_dir}/{d}/data.parquet"
                ).reset_index(drop=True)
            )
    except OSError as e:
        raise OSError(
            "No test data found. Please run get_train_and_test_data() first."
        ) from e
    return train_dict


class ModelTrainer:
    """
    ModelTrainer class is used to train a machine learning model.
    This is a Visitor class targeting the BatchSet class.
    """

    def __init__(self):
        # key is publish date, value is a list of batches
        # that have been trained
        self.trained_dates: Dict[int, Tuple[str]]

        if os.path.exists(TRAINED_DATES_PATH):
            with open(TRAINED_DATES_PATH, "rb") as f:
                self.trained_dates = pickle.load(f)
        else:
            self.trained_dates = {}
            self.dump_trained_dates()

        self._status: Dict = None
        self._task_id: str = None

    @property
    def status(self) -> Dict:
        """
        status property
        """
        return self._status

    @status.setter
    def status(self, status: Dict):
        """
        status property
        """
        self._status = status

    @property
    def task_id(self) -> str:
        """
        task_id property
        """
        return self._task_id

    @task_id.setter
    def task_id(self, task_id: str):
        """
        task_id property
        """
        self._task_id = task_id

    def update_status(
        self,
        status: str = "",
    ):
        """
        Update the status of the task.
        """
        if self._status is not None and self._task_id is not None:
            self._status[self._task_id] = f"running - step: training model: {status}"

    def dump_trained_dates(self):
        """
        Dump the trained dates to a file."""
        with open(TRAINED_DATES_PATH, "wb") as f:
            pickle.dump(self.trained_dates, f)

    def visit(self, batch_set: Any):
        """
        Visit the BatchSet class and train a machine learning model.
        """
        match batch_set:
            case BatchSet():
                self.visit_batch_set(batch_set)
            case _:
                raise TypeError(
                    f"ModelTrainer only accepts BatchSet, got {type(batch_set)}"
                )

    def visit_batch_set(self, batch_set: BatchSet):
        """
        Visit the BatchSet class and train a machine learning model.
        """
        # iterate over each publish date from oldest to newest
        # and train the model on the publish date batch
        pub_dates = batch_set.batches.keys()
        pub_dates.sort()
        for pub_date in pub_dates:
            batch = batch_set.batches[pub_date]
            # skip any publish date that has already been trained
            # we need a database of published dates and their batch of articles
            # to check if the model has already been trained
            if pub_date in self.trained_dates and batch in self.trained_dates[pub_date]:
                self.update_status(f"Skipping {pub_date} - already trained")
                continue

            for batch in batch_set.batches[pub_date]:
                # train the model on the batch
                self.update_status(f"Training {pub_date} ")
                try:
                    # get the latest training data
                    training_data = get_latest_training_data(pub_date, [TTRAIN, TTEST])
                    # train the model
                    start = datetime.now()
                    train_model(pub_date, training_data)
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} complete in {elapsed}")
                except Exception as e:
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} failed after {elapsed}: {e}")
                    continue
                # log the training
                if pub_date in self.trained_dates:
                    # add the batch articles to the articles used in prior training
                    s = self.trained_dates[pub_date]
                    s.append(batch)
                    s = list(set(s))
                    s.sort()
                    self.trained_dates[pub_date] = s
                else:
                    self.trained_dates[pub_date] = batch
                self.dump_trained_dates()


def train_model(pub_date: int, training_data) -> tuple[str, pd.DataFrame]:
    """
    train_model
        consumes the latest data in training_data and splits it 80:20 for training and
        test data.
        Trains the LoRA for the mistral model and saves it to the archive
        lora_adaptors/mistral_lora_{YYYmmddHHMM}
        returns a tuple containing the path to the new LoRa and a dataframe of test data
    """
    lora_dir = (
        f"lora_adapters/mistal_lora_{pub_date}_{datetime.now().strftime(r'%Y%m%d%H%M')}"
    )
    os.makedirs(f"./{lora_dir}", exist_ok=True)

    t_train_dataset = training_data[TTRAIN]
    t_test_dataset = training_data[TTEST]

    # Step 5: Load Model and Apply LoRA Fine-Tuning
    model = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    # Resize token embeddings if a new pad token was added
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./news_finetune_model",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        remove_unused_columns=False,  # Ensure model gets correct inputs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=t_train_dataset,
        eval_dataset=t_test_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(lora_dir)
    with open("./lora_adapters.txt", "a", encoding="utf8") as fh:
        fh.write(f"{lora_dir}\n")

    # clear the cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
