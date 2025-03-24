"""
newsies.llm
"""

from gpt4all import GPT4All
import torch

# pylint: disable=global-statement

# Load GPT4All model from local path
MODEL_PATH = "/home/mpeters/.local/share/nomic.ai/GPT4All/"
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

LLM = None


def init_llm():
    """init_llm"""
    global LLM

    if LLM is None:
        device_str = (
            f"cuda:{torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {device_str}")
        LLM = GPT4All(
            model_name=MODEL_NAME,
            model_path=MODEL_PATH,
            #    device=device_str,
            allow_download=False,
        )


def identify_themes(uri: str):
    """
    identify_themes:
      - Identify themes in a news article
    """
    init_llm()

    with open(uri, "r", encoding="utf8") as f:
        text = f.read()

    text = (
        """You are a helpful theme identifier. Please identify the themes in the following text:\n\n
    """
        + text
    )
    response = LLM.generate(text, max_tokens=250)
    return response
