"""newsies.lora_adapter"""

import os
from typing import Tuple, Union


from peft import PeftModel
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def get_latest_lora_adapter():
    """
    get_latest_lora_adapter
    """
    with open("./lora_adapters.txt", "rb") as file:
        file.seek(-2, os.SEEK_END)

        # If the file is empty, return an empty string
        if file.tell() == 0:
            return ""

        offset = -2
        whence = os.SEEK_END
        while file.read(1) != b"\n":
            try:
                file.seek(offset, whence)
            except (
                OSError
            ):  # Handle cases where the file size is smaller than the offset
                file.seek(0)  # Go to the beginning of the file
                return file.readline().decode("utf8").strip()
        return file.readline().decode("utf-8").strip()


def load_model(
    lora_dir: str = None,
) -> Tuple[PeftModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """Loads base Mistral-7B-v0.3 model and applies a LoRA adapter."""

    device_str = (
        f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device_str}")

    base_model_name = "mistralai/Mistral-7B-v0.3"

    if lora_dir is None:
        lora_dir = get_latest_lora_adapter()

    # Load tokenizer and ensure padding token exists
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = (
        AutoTokenizer.from_pretrained(base_model_name)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Load LoRA adapter
    model: PeftModel = PeftModel.from_pretrained(
        model=base_model, model_id=lora_dir, device_str=device_str
    )
    model = model.merge_and_unload()
    model.eval()  # Set to evaluation mode

    return model, tokenizer
