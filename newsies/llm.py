"""
newsies.llm
"""

from newsies.lora_adapter import load_model

# pylint: disable=global-statement

# Load GPT4All model from local path
# MODEL_PATH = "/home/mpeters/.local/share/nomic.ai/GPT4All/"
# MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

LLM = None
TOKENIZER = None


def init_llm():
    """init_llm"""
    global LLM, TOKENIZER

    if LLM is None or TOKENIZER is None:

        LLM, TOKENIZER = load_model()


#        LLM = GPT4All(
#            model_name=MODEL_NAME,
#            model_path=MODEL_PATH,
#            device=device_str,
#            allow_download=False,
#        )


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
