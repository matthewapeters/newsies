"""newsies.llm.summarize_visitor"""

import gc
import re
from typing import List, Tuple
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from newsies.ap_news.archive_visitor import ArchiveVisitor
from newsies.ap_news.archive import Archive
from newsies.ap_news.article import Article

# pylint: disable=global-statement

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_MODELS_PATH = Path.home().joinpath("mistral_models", MODEL_NAME)
MODEL = None
TOKENIZER = None


def extract_summary_blocks(text, tag_str: str = "SUMMARY"):
    """
    Extracts all content between [|SUMMARY START|] and [|SUMMARY END|] tags.
    Returns a list of matched strings.
    """
    pattern = r"\[\|" + tag_str + r" START\|\](.*?)\[\|" + tag_str + r" END\|\]"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def model_download():
    """model_download"""
    if MISTRAL_MODELS_PATH.exists():
        print(f"mistral models already downloaded to {MISTRAL_MODELS_PATH}")
    else:
        print(f"downloading {MODEL_NAME} to {MISTRAL_MODELS_PATH}")
        MISTRAL_MODELS_PATH.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=MODEL_NAME,
            allow_patterns=[
                "params.json",
                "consolidated.safetensors",
                "tokenizer.model.v3",
            ],
            local_dir=MISTRAL_MODELS_PATH,
        )


def get_model() -> Tuple:
    """get_model"""
    global MODEL
    global TOKENIZER
    if MODEL is None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

    return MODEL, TOKENIZER


class SummarizeVisitor(ArchiveVisitor):
    """SummarizeVisitor"""

    def __init__(self):
        super().__init__(
            target_type=Archive,
            step_name="Summarize Articles",
            history_path="./daily_news/apnews.com/summary_dates.pkl",
        )
        self.model, self.tokenizer = get_model()
        self.qty: int = 1  # Number of response generated by model

    @staticmethod
    def get_prompt(a: Article) -> str:
        """get_prompt"""

        return (
            """
<s>
[INST]
You are a summarization system that produces two- to three-sentence summaries of articles.

# Instructions
Summaries should be two- to -three sentences (inside [|SUMMARY START|] ... [|SUMMARY END|]).
The summaries should include one or two significant facts from the article, but should still
be terse.  Summaries should use proper names and titles whenever possible and minimize the
use of pronouns

1. A factual summary of the main events
2. A factual summary of the participating individuals and affected places or objects. Always include
    names and titles.
3. A summary of the societal, emotional, or possible future impact

## An example is below:
----------------------------
    Article Context:
    [|ITEM ID START|]abcdef0123456789 [|ITEM ID END|]
    [|ARTICLE START|]
    There is a toybox in my mother's basement that was a gift from my grandfather.
    The toybox contains three blue balls and two brown balls, and a contingent of plastic army
    men from Company Alpha.
    All the balls are different sizes, but all of the soldiers are the same size.  The toybox
    is red painted wood with brass hinges and leather handles.
    [|ARTICLE END|]


    [|SUMMARY START|]
    There are three blue balls and two brown balls in the basement toybox.
    The toybox also contains Company Alpha army men.
    [|SUMMARY END|]

    [|SUMMARY START|]
    The article mentions the author's mother and grandfather.
    [|SUMMARY END|]

    [|SUMMARY START|]
    The article about the author's toybox is rather terse and emotionless.
    [|SUMMARY END|]
----------------------------


Article Context:
----------------------------
"""
            + a.formatted
            + """
[/INST]

[|SUMMARY START|]
"""
        )

    @staticmethod
    def qa_prompt(summary: str) -> str:
        """qa_prompt"""
        return (
            """
    <s>
    [INST]
    You are a question-and-answer generation system.  Generate a specific, detailed question
    regarding the events, entities and impacts of the provided summary.  Generate answers
    that leverage as many meaningful details from the provided summary as necessary to fully
    answer the question.

    An example follows:
    ----------------------------

    [|SUMMARY START|]
    There are three blue balls and two brown balls in the basement toybox. The toybox also
    contains plastic army men from Company Alpha.
    [|SUMMARY END|]

    [|QUESTION START|]
    How many balls are in the basement toybox?
    [|QUESTION END|]

    [|ANSWER START|]
    There are five balls in the basement toybox: two brown and three blue.
    [|ANSWER END|]

    [/INST]

    [|SUMMARY START|]
    """
            + summary
            + """
    [|SUMMARY END|]

    [|QUESTION START|]
    """
        )

    def gen_qa(self, summary: str, qty: int = 1):
        """gen_qa"""
        input_ids = self.tokenizer(summary, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to("cuda")
        return self.model.generate(
            input_ids,
            max_length=2048,
            temperature=0.40,
            top_p=0.98,
            do_sample=True,
            num_return_sequences=qty,
        )

    def generate_summary(self, prompt: str, qty: int):
        """generate_summary"""
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to("cuda")
        resp = self.model.generate(
            input_ids,
            max_length=2048,
            temperature=0.7,
            top_p=0.98,
            do_sample=True,
            num_return_sequences=qty,
        )
        output = []
        for i in range(qty):
            output.append(self.tokenizer.decode(resp[i]))
        return output

    def visit_archive(self, archive: Archive):
        """visit_archive

        required by abstract base class
        """
        # Prompt the MistralAI-Instruct model to generate three summaries
        # of the article, each focussed on separate dimensions of the story:
        #   Events
        #   Participating Facts / NERs
        #   Impacts (societal, emotional or future)
        #
        # These will be used to prompt for Questions and Answers used in training
        # the LLM.
        #
        # The summaries, as well as the original article will be used in training
        # to provide semantic mapping to the article item_id, along with the Q&A
        #
        article: Article = None
        summaries: List[str] = None

        # get articles from Archive

        # iterate over articles
        for pub_date, item_id in [
            (d, i) for d in article.publish_dates for i in archive.by_publish_date[d]
        ]:
            article = archive.get_article(item_id)
            if self.step_name in article.pipelines:
                continue
            print(f"Generate Summaries for {pub_date}: {item_id}")
            summaries = [
                s
                for o in self.generate_summary(self.get_prompt(article), self.qty)
                for s in extract_summary_blocks(o, "SUMMARY")[-3:]
            ]
            print(f"Generate Questions for {pub_date}: {item_id}")
            for s in summaries:
                response = self.gen_qa(self.qa_prompt(s), self.qty)
                for r in response:
                    rr = self.tokenizer.decode(r)
                    q = extract_summary_blocks(rr, "QUESTION")[-1]
                    a = extract_summary_blocks(rr, "ANSWER")[-1]
                    article.add_summary_and_qa(s, q, a)
            article.pipelines[self.step_name] = datetime.now().isoformat()
            article.pickle()
            print(f"{pub_date}: {item_id} Complete")
