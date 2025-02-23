from gpt4all import GPT4All

from newsies.chromadb_client import ChromaDBClient
import torch

# Load GPT4All model from local path
MODEL_PATH = "/home/mpeters/.local/share/nomic.ai/GPT4All/"
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
device_str = (
    f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
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
    with open(uri, "r") as f:
        text = f.read()

    text = (
        """You are a helpful theme identifier. Please identify the themes in the following text:\n\n
    """
        + text
    )
    response = LLM.generate(text, max_tokens=250)
    return response


def identify_entities(uri: str):
    """
    identify_entities:
      - Identify entities in a news article
    """
    with open(uri, "r") as f:
        text = f.read()

    text = (
        """You are a helpful entity identifier. Please identify the entities in the following text:\n\n
    """
        + text
    )
    response = LLM.generate(text, max_tokens=250)
    return response
