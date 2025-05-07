"""
newsies.llm.model_trainer
"""

from datetime import datetime
from typing import Dict, List
import gc
import os
import time
import shutil

from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    logging as hf_logging,
)
from transformers.utils import logging as hfu_logging

from newsies.llm.batch_set import BatchSet
from newsies.visitor import Visitor

# pylint: disable=dangerous-default-value, broad-exception-caught

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_logging.set_verbosity_error()
hfu_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

_TRAINED_DATES_PATH = "./training_data/trained_dates.pkl"
_BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

TRAIN = "train"
TEST = "test"
TTRAIN = "token_train"
TTEST = "token_test"
_TRAIN_DATA_TYPES = [TRAIN, TEST, TTRAIN, TTEST]

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
            articles: List[str] = list(
                set(a for batch in batch_set.batches[pub_date] for a in batch)
            )
            articles.sort()
            article_count = len(articles)
            # in this context, a batch is all of the articles for a publish date
            # we prepend the article count to the sorted list of articles to quickly
            # identify potential changes in the publish date corpus
            batch_id = f"{article_count}: {','.join(articles)}"

            # skip any publish date that has already been trained
            # we need a database of published dates and their batch of articles
            # to check if the model has already been trained

            # eventually, we need to add the commented out code - we might discover new articles for
            # the same publish date, so we need to check if the batch_id is the same
            if pub_date in self.history:  # and batch_id == self.history[pub_date]:
                self.update_status(f"Skipping {pub_date} - already trained")
                continue

            # train the model on the batch
            self.update_status(f"training {pub_date}")
            start = datetime.now()
            print(
                f"{NEWSIES}{TRAINING} start training "
                f"{pub_date} at {start} -- {article_count} articles"
            )
            try:
                wait_for_cuda_memory(target_gb=12.0, max_wait_sec=180)
                log_gpu_memory("Before getting trainer and model")
                try:
                    t_train_dataset, t_eval_dataset = get_train_and_test_data(pub_date)
                except OSError as e:
                    self.update_status(
                        f"{WARN} Error getting training data for {pub_date}: {e}"
                    )
                    continue
                trainer, model, tokenizer = get_trainer(t_train_dataset, t_eval_dataset)
                log_gpu_memory("After get_trainer() before train_model()")
                train_model(trainer)
                log_gpu_memory("After training")
                save_lora_adapter(pub_date, model)
                log_gpu_memory("After saving LoRA")
                maybe_merge_adapters(pub_date, merge_threshold=5)
                log_gpu_memory("After Maybe Merge")
                cleanup(
                    model,
                    tokenizer,
                    trainer,
                )
                log_gpu_memory("After Cleanup")
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{OK} {pub_date} complete in {elapsed}"
                self.history[pub_date] = batch_id
                self.dump_history()
            except TimeoutException as te:
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{FAIL} {pub_date} failed after {elapsed}: {te}"
                raise te
            except MemoryError as me:
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{FAIL} {pub_date} failed after {elapsed}: {me}"
                raise me
            except torch.OutOfMemoryError as ome:
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{FAIL} {pub_date} failed after {elapsed}: {ome}"
                raise ome
            except RuntimeError as re:
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{FAIL} {pub_date} failed after {elapsed}: {re}"
                raise re
            except Exception as e:
                end = datetime.now()
                elapsed = end - start
                _msg = f"{NEWSIES}{WARN} {pub_date} failed after {elapsed}: {e}"
            finally:
                print(_msg)
                self.update_status(_msg)


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
    print(f"{BOOKS}{OK} Training data loaded and validated.")
    return t_train_dataset, t_eval_dataset


def get_trainer(
    t_train_dataset, t_eval_dataset
) -> tuple[Trainer, torch.nn.Module, AutoTokenizer]:
    """
    get_trainer
        Safely builds and returns a Trainer object, along with the model and tokenizer.
    """

    # get the model with the latest LoRA adapter
    model = load_base_model_with_lora(training_mode=True)
    log_gpu_memory("in get_trainer() after loading model (training_mode=True)")

    # get the tokenizer
    model_path = _BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Required for collator

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Select batch size
    batch_sizes = [24, 16, 12, 8, 4, 2, 1]
    selected_batch_size = auto_select_batch_size(batch_sizes)

    training_args = TrainingArguments(
        output_dir="./news_finetune_model",
        per_device_train_batch_size=selected_batch_size,
        num_train_epochs=3,
        logging_steps=10000,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        disable_tqdm=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=t_train_dataset,
        eval_dataset=t_eval_dataset,
        data_collator=data_collator,
    )

    return trainer, model, tokenizer


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
        print(f"{TRAINING}{OK} Training complete.")
    except Exception as e:
        print(f"️{TRAINING}{WARN} Training failed: {e}")
        raise


def save_lora_adapter(pub_date, model) -> None:
    """save_lora_adapter"""
    try:
        lora_dir = f"lora_adapters/mistral_lora_{pub_date}_{datetime.now().strftime(r'%Y%m%d%H%M')}"
        os.makedirs(f"./{lora_dir}", exist_ok=True)
        model.save_pretrained(lora_dir)
        with open("./lora_adapters.txt", "a", encoding="utf8") as fh:
            fh.write(f"./{lora_dir}\n")
        print(f"{DISK}{OK} LoRA adapter saved to {lora_dir}")
    except Exception as e:
        print(f"️{DISK}{WARN} Error saving LoRA adapter: {e}")
        raise


def cleanup(model, tokenizer, trainer) -> None:
    """
    cleanup
    """
    # Remove the model from GPU memory
    if model:
        try:
            model.cpu()  # Move model to CPU to free GPU memory
            del model
        except Exception as e:
            print(f"{CLEAN}{WARN} Error deleting model: {e}")
    if tokenizer:
        try:
            del tokenizer
        except Exception as e:
            print(f"{CLEAN}{WARN} Error deleting tokenizer: {e}")
    if trainer:
        try:
            if hasattr(trainer, "model"):
                trainer.model.cpu()  # Move trainer model to CPU to free GPU memory
            trainer.model = None
            trainer.traine_dataset = None
            trainer.eval_dataset = None
            trainer.data_collator = None
            trainer.args = None
            trainer.processing_class = None
            del trainer
        except Exception as e:
            print(f"{CLEAN}{WARN} Error deleting trainer: {e}")
    gc.collect()
    time.sleep(1)
    gc.collect()
    # clear the cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"{CLEAN}{OK} Cleanup complete.")


def maybe_merge_adapters(pub_date: int, merge_threshold: int = 5) -> None:
    """
    maybe_merge_adapters
        Checks how many progressive LoRA adapters have been trained.
        If the number meets or exceeds merge_threshold, it merges the latest LoRA adapter
        into the base model and saves it as a new merged model.
        Also updates 'latest_merged_model.txt' with the path to the latest merged model.
    """
    if not os.path.exists("./lora_adapters.txt"):
        print(f"{INFO} No lora_adapters.txt file found. Skipping merge.")
        return

    with open("./lora_adapters.txt", "r", encoding="utf8", newline="") as fh:
        adapters = [line.strip() for line in fh if line.strip()]

    if len(adapters) % merge_threshold != 0:
        print(
            f"️{PACKAGE}{WAIT} Only {len(adapters)} adapters found. "
            f"Merge modulo is {merge_threshold}. Skipping merge."
        )
        return

    # Step 1: Load model with the latest LoRA adapter
    model = load_base_model_with_lora(training_mode=False)
    log_gpu_memory("after loading model (training_mode=False)")

    # Step 2: Save the merged model
    merged_base = "merged_models"
    merged_dir = f"{merged_base}/merged_model_{pub_date}_{datetime.now().strftime(r'%Y%m%d%H%M')}"
    os.makedirs(merged_dir, exist_ok=True)

    # Remove old merged models:
    for old_path in os.listdir(merged_base):
        if old_path.startswith("merged_model_"):
            shutil.rmtree(os.path.join(merged_base, old_path))

    # Save the merged model
    model.save_pretrained(merged_dir)

    # Step 3: Save the path of the latest merged model
    with open("./latest_merged_model.txt", "w", encoding="utf8", newline="") as f:
        f.write(merged_dir)

    print(f"{PACKAGE}{OK} Successfully merged and saved model to {merged_dir}")
    cleanup(model, None, None)


class TimeoutException(Exception):
    """TimeoutException"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        time.sleep(1)
        gc.collect()

        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        available = total - reserved - allocated
        available_gb = available / (1024**3)

        if available_gb >= target_gb:
            print(
                f"{CLEAN}{OK} Sufficient GPU memory restored: {available_gb:.2f} GiB available."
            )
            break

        elapsed = time.time() - start
        if elapsed > max_wait_sec:
            raise TimeoutException(
                f"️{CLEAN}{WARN} Timeout: Only {available_gb:.2f} "
                f"GiB available after {max_wait_sec} sec."
            )

        print(
            f"{CLEAN}{WAIT} Waiting for GPU memory to flush... {available_gb:.2f} GiB available"
        )
        time.sleep(2)  # Let CUDA finish releasing memory


def log_gpu_memory(context=""):
    """log_gpu_memory"""
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    free = total - reserved - allocated
    print(
        f"{SEARCH}{INFO} [{context}] Allocated: {allocated / 1e9:.2f} GB | "
        f"Reserved: {reserved / 1e9:.2f} GB | Free: {free / 1e9:.2f} GB"
    )


def auto_select_batch_size(possible_batch_sizes: list[int]) -> int:
    """
    Automatically selects the largest batch size that fits in GPU memory.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(f"{TRAINING}{WARN} No CUDA device available. Defaulting to batch size 1.")
        return 1

    torch.cuda.empty_cache()
    max_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = max_memory - reserved_memory - allocated_memory

    print(
        f"{TRAINING}{SEARCH} Available GPU memory: {available_memory / (1024**3):.2f} GiB"
    )

    # Start from largest batch size and work down
    for batch_size in sorted(possible_batch_sizes, reverse=True):
        # Assume roughly 0.5 GiB per batch element (adjustable based on your model size!)
        expected_usage = batch_size * 0.5 * (1024**3)  # 0.5 GB per sample

        if expected_usage < available_memory * 0.8:  # Use 80% of available
            print(f"{TRAINING}{OK} Selected batch size: {batch_size}")
            return batch_size

    print(f"️️{TRAINING}{WARN} Defaulting to batch size 1.")
    return 1


def validate_dataset(ds):
    """
    Validate the dataset to ensure it has the required columns and that they are not empty.
    """
    required_columns = {"input_ids", "attention_mask", "labels"}
    assert set(ds.column_names).issuperset(
        required_columns
    ), f"️{BOOKS}{WARN} Dataset columns missing: {required_columns - set(ds.column_names)}"
    for idx in range(min(10, len(ds))):  # Just sample 10 rows for sanity check
        row = ds[idx]
        for col in required_columns:
            assert col in row, f"️{BOOKS}{WARN} Missing column {col} in row {idx}"
        assert len(row["input_ids"]) > 0, f"{BOOKS}{WARN} Empty input_ids at row {idx}"


def get_latest_training_data(
    pub_date: int, types: List[str] = _TRAIN_DATA_TYPES
) -> Dict[str, pd.DataFrame]:
    """
    get_latest_train_data
        Retrieves the most recent representation of training data for a publish date.
    """
    base_dir = "./train_test"
    if isinstance(pub_date, int):
        base_dir = f"./{base_dir}/{pub_date:04d}"
    else:
        base_dir = f"./{base_dir}/{pub_date}"
    if not os.path.exists(base_dir):
        with open("missing_training_data.txt", "a", encoding="utf8") as fh:
            fh.write(f"{base_dir}\n")
        msg = (
            f"️{BOOKS}{FAIL} No training data found for {base_dir}. "
            "Please run get_train_and_test_data() first."
        )
        print(msg)
        raise OSError(msg)
    dirs = os.listdir(base_dir)
    if len(dirs) == 0:
        raise OSError("️No test data found. Please run get_train_and_test_data() first.")
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


def load_base_model_with_lora(training_mode: bool = True) -> torch.nn.Module:
    """
    Loads the base model and applies the latest LoRA adapter (if any).

    If `training_mode=True`:
        - merge the previous adapter if it's not the first one,
        - then apply a new adapter.

    If `training_mode=False`:
        - merge the most recent adapter into the base model (for export), using CPU to avoid OOM.
    """
    device_map = "cuda" if training_mode else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL_NAME, torch_dtype=torch.float16, device_map=device_map
    )

    adapters = []
    if os.path.exists("./lora_adapters.txt"):
        with open("./lora_adapters.txt", "r", encoding="utf8") as fh:
            adapters = [line.strip() for line in fh if line.strip()]

    if adapters:
        last_lora_path = adapters[-1]
        peft_config = PeftConfig.from_pretrained(last_lora_path)

        if (
            peft_config.base_model_name_or_path
            and peft_config.base_model_name_or_path != _BASE_MODEL_NAME
        ):
            raise RuntimeError(
                f"Incompatible adapter. Expected base: {_BASE_MODEL_NAME}, "
                f"got: {peft_config.base_model_name_or_path}"
            )

        print(f"{TRAINING}{SEARCH} Loading latest LoRA adapter: {last_lora_path}")
        model = PeftModel.from_pretrained(
            model, last_lora_path, torch_dtype=torch.float16
        )

        if not training_mode:
            print(f"{PACKAGE}{INFO} Merging adapter for inference on CPU...")
            if not isinstance(model, PeftModel):
                raise RuntimeError(
                    f"Expected PeftModel, got {type(model)}. "
                    "Check if the model is already merged."
                )
            model = model.merge_and_unload()
            gc.collect()
            time.sleep(1)
            gc.collect()
            torch.cuda.empty_cache()
            return model

        if len(adapters) > 1 and hasattr(model, "merge_and_unload"):
            print(
                f"{PACKAGE}{INFO} Merging previous LoRA into model (before applying new)..."
            )
            model = model.merge_and_unload()
            torch.cuda.empty_cache()

    print(f"{TRAINING}{INFO} Applying new LoRA adapter for training...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    gc.collect()
    return model
