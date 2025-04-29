"""
newsies.llm.model_trainer
"""

from datetime import datetime
from typing import Dict, List
import gc
import os
import time

from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from newsies.llm.batch_set import BatchSet
from newsies.visitor import Visitor

# pylint: disable=dangerous-default-value, broad-exception-caught

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_TRAINED_DATES_PATH = "./training_data/trained_dates.pkl"
_BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
_TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]


def auto_select_batch_size(possible_batch_sizes: list[int]) -> int:
    """
    Automatically selects the largest batch size that fits in GPU memory.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("No CUDA device available. Defaulting to batch size 1.")
        return 1

    torch.cuda.empty_cache()
    max_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = max_memory - reserved_memory - allocated_memory

    print(f"Available GPU memory: {available_memory / (1024**3):.2f} GiB")

    # Start from largest batch size and work down
    for batch_size in sorted(possible_batch_sizes, reverse=True):
        # Assume roughly 0.5 GiB per batch element (adjustable based on your model size!)
        expected_usage = batch_size * 0.5 * (1024**3)  # 0.5 GB per sample

        if expected_usage < available_memory * 0.8:  # Use 80% of available
            print(f"Selected batch size: {batch_size}")
            return batch_size

    print("Defaulting to batch size 1.")
    return 1


def validate_dataset(ds):
    """
    Validate the dataset to ensure it has the required columns and that they are not empty.
    """
    required_columns = {"input_ids", "attention_mask", "labels"}
    assert set(ds.column_names).issuperset(
        required_columns
    ), f"Dataset columns missing: {required_columns - set(ds.column_names)}"
    for idx in range(min(10, len(ds))):  # Just sample 10 rows for sanity check
        row = ds[idx]
        for col in required_columns:
            assert col in row, f"Missing column {col} in row {idx}"
        assert len(row["input_ids"]) > 0, f"Empty input_ids at row {idx}"


def get_latest_training_data(
    pub_date: int, types: List[str] = _TRAIN_DATA_TYPES
) -> Dict[str, pd.DataFrame]:
    """
    get_latest_train_data
    """
    base_dir = "./train_test"
    if isinstance(pub_date, int):
        base_dir = f"./{base_dir}/{pub_date:04d}"
    else:
        base_dir = f"./{base_dir}/{pub_date}"
    dirs = os.listdir(base_dir)
    if len(dirs) == 0:
        raise OSError("No test data found. Please run get_train_and_test_data() first.")
    dirs.sort()
    latest_dir = dirs[-1]
    train_dict: Dict[str, Dataset] = {}

    d: str = "unset"
    try:
        for d in types:
            train_dict[d] = Dataset.from_pandas(
                pd.read_parquet(
                    f"{base_dir}/{latest_dir}/{d}/data.parquet"
                ).reset_index(drop=True)
            )
    except OSError as e:
        raise OSError(
            f"No data found for {base_dir}/{pub_date}/{d}. "
            "Please run get_train_and_test_data() first."
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
                    wait_for_cuda_memory(target_gb=12.0, max_wait_sec=180)
                    try:
                        t_train_dataset, t_eval_dataset = get_train_and_test_data(
                            pub_date
                        )
                    except OSError as e:
                        self.update_status(
                            f"Error getting training data for {pub_date}: {e}"
                        )
                        continue
                    trainer, model, base_model, tokenizer = get_trainer(
                        t_train_dataset, t_eval_dataset
                    )
                    train_model(trainer)
                    save_model(pub_date, model)
                    maybe_merge_adapters(merge_threshold=5)
                    cleanup(
                        model,
                        base_model,
                        tokenizer,
                        trainer,
                    )
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} complete in {elapsed}")
                    self.history[pub_date] = batch_id
                    self.dump_history()
                except Exception as e:
                    end = datetime.now()
                    elapsed = end - start
                    self.update_status(f"{pub_date} failed after {elapsed}: {e}")


def get_train_and_test_data(pub_date: int) -> Dict[str, pd.DataFrame]:
    """
    get_train_and_test_data
    """
    # get the latest training data
    training_data = get_latest_training_data(pub_date, [TTRAIN, TTEST])

    t_train_dataset = training_data[TTRAIN]
    t_eval_dataset = training_data[TTEST]

    # remove unwanted columns
    t_train_dataset = t_train_dataset.remove_columns(["pubdate", "batch"])
    t_eval_dataset = t_eval_dataset.remove_columns(["pubdate", "batch"])

    # Ensure the dataset includes both input_ids and labels
    t_train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    t_eval_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    validate_dataset(t_train_dataset)
    validate_dataset(t_eval_dataset)
    return t_train_dataset, t_eval_dataset


def get_trainer(t_train_dataset, t_eval_dataset) -> ModelTrainer:
    """
    get_trainer
        Returns a new instance of the ModelTrainer class.
    """
    # Decide which model to load
    model_path = _BASE_MODEL_NAME

    # Always load the model first
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    model = base_model

    # Apply previous LoRA if it exists
    if os.path.exists("./lora_adapters.txt"):
        with open("./lora_adapters.txt", "r", encoding="utf8") as fh:
            adapters = [line.strip() for line in fh if line.strip()]
        if adapters:
            last_lora_path = adapters[-1]
            print(f"Loading previous LoRA adapter: {last_lora_path}")
            model = PeftModel.from_pretrained(
                model, last_lora_path, torch_dtype=torch.float16
            )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainer: Trainer = None

    batch_sizes = [24, 16, 12, 8, 4, 2, 1]
    selected_batch_size = auto_select_batch_size(batch_sizes)

    try:
        training_args = TrainingArguments(
            output_dir="./news_finetune_model",
            per_device_train_batch_size=selected_batch_size,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # because it's causal LM (not masked LM)
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=t_train_dataset,
            eval_dataset=t_eval_dataset,
            data_collator=data_collator,
        )
    except Exception as e:
        print(f"Error during Trainer initialization: {e}")
        raise

    return trainer, model, base_model, tokenizer


def train_model(trainer) -> tuple[str, pd.DataFrame]:
    """
    train_model
        consumes the latest data in training_data and splits it 80:20 for training and
        test data.
        Trains the LoRA for the mistral model and saves it to the archive
        lora_adapters/mistral_lora_{YYYYmmddHHMM}
        returns a tuple containing the path to the new LoRa and a dataframe of test data
    """
    try:
        trainer.train()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Training failed: {e}")
        raise


def save_model(pub_date, model) -> None:
    """save_model"""
    try:
        lora_dir = f"lora_adapters/mistral_lora_{pub_date}_{datetime.now().strftime(r'%Y%m%d%H%M')}"
        os.makedirs(f"./{lora_dir}", exist_ok=True)
        model.save_pretrained(lora_dir)
        with open("./lora_adapters.txt", "a", encoding="utf8") as fh:
            fh.write(f"./{lora_dir}\n")
    except Exception as e:
        print(f"Error saving LoRA adapter: {e}")
        raise


def cleanup(model, base_model, tokenizer, trainer) -> None:
    """
    cleanup
    """
    # Remove the model from GPU memory
    try:
        del model
    except Exception as e:
        print(f"Error deleting model: {e}")
    try:
        del base_model
    except Exception as e:
        print(f"Error deleting base model: {e}")
    try:
        del tokenizer
    except Exception as e:
        print(f"Error deleting tokenizer: {e}")
    try:
        del trainer
    except Exception as e:
        print(f"Error deleting trainer: {e}")
    gc.collect()
    # clear the cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Cleanup complete.")


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

    del model
    gc.collect()

    # Step 8: Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wait_for_cuda_memory(target_gb: float = 10.0, max_wait_sec: int = 60):
    """
    Wait for available CUDA memory to reach a target threshold before continuing.
    This helps mitigate asynchronous memory release issues.
    """
    if not torch.cuda.is_available():
        return

    start = time.time()
    while True:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        available = total - reserved - allocated
        available_gb = available / (1024**3)

        if available_gb >= target_gb:
            print(
                f"✅ Sufficient GPU memory restored: {available_gb:.2f} GiB available."
            )
            break

        elapsed = time.time() - start
        if elapsed > max_wait_sec:
            print(
                f"⚠️ Timeout: Only {available_gb:.2f} GiB available after {max_wait_sec} sec."
            )
            break

        print(f"⏳ Waiting for GPU memory to flush... {available_gb:.2f} GiB available")
        time.sleep(2)  # Let CUDA finish releasing memory
