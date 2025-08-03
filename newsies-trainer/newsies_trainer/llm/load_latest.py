"""
newsies.llm.load_latest
"""

import gc
import os
import time

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import torch
from transformers import (
    AutoModelForCausalLM,
)
from .specs import (
    _BASE_MODEL_NAME,
    INFO,
    PACKAGE,
    SEARCH,
    TRAINING,
)


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
