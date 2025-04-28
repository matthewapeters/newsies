"""
newsies.llm.model_trainer
"""

from datetime import datetime
from typing import Dict, List
import os

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from newsies.llm.batch_set import BatchSet
from newsies.visitor import Visitor

# pylint: disable=dangerous-default-value, broad-exception-caught

_TRAINED_DATES_PATH = "./training_data/trained_dates.pkl"
_BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
_TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]


class FastEncodedDataset(Dataset):
    """
    FastEncodedDataset class is used to load the dataset into memory.
    """

    def __init__(self, dataset):
        super().__init__()
        # Actually load the tensors into memory

        def ensure_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x)

        self.input_ids = ensure_tensor(dataset["input_ids"])
        self.attention_mask = ensure_tensor(dataset["attention_mask"])
        self.labels = ensure_tensor(dataset["labels"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def get_latest_training_data(
    pub_date: int, types: List[str] = _TRAIN_DATA_TYPES
) -> Dict[str, pd.DataFrame]:
    """
    get_latest_train_data
    """
    try:
        base_dir = "./train_test"
        if isinstance(pub_date, int):
            base_dir = f"./{base_dir}/{pub_date:04d}"
        else:
            base_dir = f"./{base_dir}/{pub_date}"
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


class ModelTrainer(Visitor):
    """
    ModelTrainer class is used to train a machine learning model.
    This is a Visitor class targeting the BatchSet class.
    """

    def __init__(self):
        # key is publish date, value is a list of batches
        # that have been trained
        super().__init__(
            BatchSet,
            _TRAINED_DATES_PATH,
            "ModelTrainer",
        )

    def visit_batch_set(self, batch_set: BatchSet):
        """
        Visit the BatchSet class and train a machine learning model.
        """
        # iterate over each publish date from oldest to newest
        # and train the model on the publish date batch
        pub_dates = list(batch_set.batches.keys())
        pub_dates.sort()
        for pub_date in pub_dates:
            batches = batch_set.batches[pub_date]
            for batch in batches:
                batch_id = list(set(batch))
                batch_id.sort()
                batch_id = ",".join(batch_id)

                # skip any publish date that has already been trained
                # we need a database of published dates and their batch of articles
                # to check if the model has already been trained
                if pub_date in self.history and batch_id in self.history[pub_date]:
                    self.update_status(f"Skipping {pub_date} - already trained")
                    continue

                # train the model on the batch
                self.update_status(f"training {pub_date} ")
                start = datetime.now()
                try:
                    # get the latest training data
                    training_data = get_latest_training_data(pub_date, [TTRAIN, TTEST])
                    # train the model
                    train_model(pub_date, training_data)
                    maybe_merge_adapters(merge_threshold=5)
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} complete in {elapsed}")
                    self.history[pub_date] = batch_id
                    self.dump_history()
                except Exception as e:
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} failed after {elapsed}: {e}")


def train_model(pub_date: int, training_data) -> tuple[str, pd.DataFrame]:
    """
    train_model
        consumes the latest data in training_data and splits it 80:20 for training and
        test data.
        Trains the LoRA for the mistral model and saves it to the archive
        lora_adapters/mistral_lora_{YYYYmmddHHMM}
        returns a tuple containing the path to the new LoRa and a dataframe of test data
    """
    lora_dir = f"lora_adapters/mistral_lora_{pub_date}_{datetime.now().strftime(r'%Y%m%d%H%M')}"

    t_train_dataset = training_data[TTRAIN]
    t_eval_dataset = training_data[TTEST]

    # Ensure the dataset includes both input_ids and labels
    t_train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    t_eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    # remove unwanted columns
    t_train_dataset = t_train_dataset.remove_columns(["pubdate", "batch"])
    t_eval_dataset = t_eval_dataset.remove_columns(["pubdate", "batch"])

    # wrap datasets
    train_dataset = FastEncodedDataset(t_train_dataset)
    eval_dataset = FastEncodedDataset(t_eval_dataset)

    # Step 1: Decide which model to load
    model_path = _BASE_MODEL_NAME
    if os.path.exists("./latest_merged_model.txt"):
        with open("./latest_merged_model.txt", "r", encoding="utf8") as f:
            merged_model_path = f.read().strip()
        if merged_model_path and os.path.exists(merged_model_path):
            print(f"Loading latest merged model from: {merged_model_path}")
            model_path = merged_model_path
        else:
            print("Merged model path invalid. Falling back to base model.")

    # Step 2: Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    # Resize token embeddings if a new pad token was added
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Step 3: Load previous LoRA adapter if it exists
    if os.path.exists("./lora_adapters.txt"):
        with open("./lora_adapters.txt", "r", encoding="utf8") as fh:
            adapters = [line.strip() for line in fh if line.strip()]
        if adapters:
            last_lora_path = adapters[-1]
            print(f"Loading previous LoRA adapter: {last_lora_path}")
            model = PeftModel.from_pretrained(model, last_lora_path)

    # Step 4: Apply new LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Step 5: Train
    trainer: Trainer = None
    try:
        training_args = TrainingArguments(
            output_dir="./news_finetune_model",
            per_device_train_batch_size=1,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            fp16=True,
            optim="adamw_torch",
            label_names=["labels"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
    except Exception as e:
        print(f"Error during Trainer initialization: {e}")
        raise
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    # Step 6: Save new LoRA
    try:
        os.makedirs(f"./{lora_dir}", exist_ok=True)
        model.save_pretrained(lora_dir)
        with open("./lora_adapters.txt", "a", encoding="utf8") as fh:
            fh.write(f"./{lora_dir}\n")
    except Exception as e:
        print(f"Error saving LoRA adapter: {e}")
        raise

    # clear the cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return lora_dir, t_eval_dataset


def maybe_merge_adapters(merge_threshold: int = 5) -> None:
    """
    maybe_merge_adapters
        Checks how many progressive LoRA adapters have been trained.
        If the number meets or exceeds merge_threshold, it merges the latest LoRA adapter
        into the base model and saves it as a new merged model.
        Also updates 'latest_merged_model.txt' with the path to the latest merged model.
    """
    if not os.path.exists("./lora_adapters.txt"):
        print("No lora_adapters.txt file found. Skipping merge.")
        return

    with open("./lora_adapters.txt", "r", encoding="utf8", newline="") as fh:
        adapters = [line.strip() for line in fh if line.strip()]

    if len(adapters) < merge_threshold:
        print(
            f"Only {len(adapters)} adapters found. "
            f"Merge threshold is {merge_threshold}. Skipping merge."
        )
        return

    # Step 1: Decide which model to load
    model_path = _BASE_MODEL_NAME
    if os.path.exists("./latest_merged_model.txt"):
        with open("./latest_merged_model.txt", "r", encoding="utf8", newline="") as f:
            merged_model_path = f.read().strip()
        if merged_model_path and os.path.exists(merged_model_path):
            print(f"Loading latest merged model from: {merged_model_path}")
            model_path = merged_model_path
        else:
            print("Merged model path invalid. Falling back to base model.")

    # Step 2: Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Step 3: Load only the latest adapter
    last_adapter = adapters[-1]
    print(f"Applying latest adapter: {last_adapter}")
    model = PeftModel.from_pretrained(model, last_adapter)

    # Step 4: Merge the adapter into the model
    model = model.merge_and_unload()

    # Step 5: Save the merged model
    merged_dir = f"merged_models/merged_model_{datetime.now().strftime(r'%Y%m%d%H%M')}"
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)

    # Step 6: Save the path of the latest merged model
    with open("./latest_merged_model.txt", "w", encoding="utf8", newline="") as f:
        f.write(merged_dir)

    print(f"Successfully merged and saved model to {merged_dir}")

    # Step 7: Clear the lora_adapters.txt file
    open("./lora_adapters.txt", "w", encoding="utf8", newline="").close()
    print("Cleared lora_adapters.txt after merge.")

    # Step 8: Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
