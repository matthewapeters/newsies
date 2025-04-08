"""
newsies.llm.question_generator
"""

from typing import Any
import os
from datetime import datetime

import torch
from torch.multiprocessing import set_start_method

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    #    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    # TrainingArguments,
    #    Trainer,
    #    BatchEncoding,
)
import pandas as pd

# import datasets

# from newsies.ap_news.archive import get_archive, Archive
from newsies.llm import BatchSet

# pylint: disable=broad-exception-caught


class QuestionGenerator:
    """
    QuestionGenerator creates training questions for articles in BatchSets
    """

    def __init__(self, number_of_questions: int = 3, batch_size: int = 8):
        self.number_of_questions = number_of_questions
        self.batch_size = batch_size

    def visit(self, o: Any):
        """
        visit
        """
        match o:
            case BatchSet():
                o.accept(self)
            case _:
                raise TypeError(
                    f"QuestionGenerator only accepts BatchSet, got {type(o)}"
                )

    def visit_batch_set(self, batch_set: BatchSet):
        """
        visit_batch_set
        """
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

        # Model & Tokenizer Initialization
        base_model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )

        # Enable torch.compile() for speedup (if available)
        if torch.__version__ >= "2.0":
            model = torch.compile(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Initialize HF Pipeline
        # qa_generator = pipeline(
        #    "text2text-generation",
        #    model=model,
        #    tokenizer=tokenizer,
        # )

        def genereate_question_factory(batch_set_idx: int):
            """ "
            generate_question_factory
            will generate questions and save dataset as parquet file
            using batch_id for path
            """

            # Define processing function for batch generation
            def generate_questions(batch, indices):
                batch_prompts = batch["prompt"]

                # Tokenize batch (explicit tokenization since we're using .generate())
                batch_inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True, truncation=True
                ).to(device)

                # Compute batch ID from indices
                batch_id: int = (
                    indices[0] // self.batch_size
                )  # First index determines batch
                with torch.no_grad():  # disable gradient computation
                    # with torch.amp.autocast(device_type=str(device)):  # use mixed precision
                    # Generate Questions using Hugging Face Pipeline (Batch Mode)
                    batch_questions_encoded = model.generate(
                        **batch_inputs,
                        max_length=100,
                        # Generate n questions per prompt
                        num_return_sequences=self.number_of_questions,
                        do_sample=True,  # Introduce randomness for variation
                        temperature=0.7,  # Adjust temperature for diversity
                        top_p=0.9,  # Nucleus sampling for more natural responses
                    )

                batch_questions_encoded = batch_questions_encoded.cpu()

                batch_questions = tokenizer.batch_decode(
                    batch_questions_encoded, skip_special_tokens=True
                )
                # Format questions properly with prompt and context for model training
                formatted_questions = [
                    [
                        (
                            "for the next question, return the 'section', "
                            "the 'headline', and the 'URI'\n"
                            f"context section: {batch['section']}\n"
                            f"context uri: {batch['uri']}\n ontext headline: {batch['headline']}\n"
                            f"context article: {batch['doc']}\n"
                            f"question: '{batch_questions[i+ii]}'"
                        )
                        for ii in range(0, self.number_of_questions)
                    ]
                    for i in range(0, len(batch_questions), self.number_of_questions)
                ]
                batch["train_question"] = formatted_questions

                # these are just the questions without prompt and context.
                # necessary for testing the model
                unformatted_questions = [
                    [
                        (
                            "for the next question, return the 'section', "
                            "the 'headline', and the 'URI'\n"
                            f"question: '{batch_questions[i+ii]}'"
                        )
                        for ii in range(0, self.number_of_questions)
                    ]
                    for i in range(0, len(batch_questions), self.number_of_questions)
                ]
                batch["question"] = unformatted_questions
                batch = pd.DataFrame(dict(batch))
                batch = batch.explode("train_question").explode("question")

                p = f"./training_data/{batch_set_idx:04d}"
                os.makedirs(p, exist_ok=True)

                # Save Each Batch Immediately to Parquet
                save_qa_to_parquet(
                    batch,
                    file_path=f"{p}/qa_dataset_batch_{batch_id:04d}.parquet",
                )

                # 🔥 Free GPU memory manually
                del batch_inputs, batch_questions_encoded  # Delete tensors
                torch.cuda.empty_cache()  # Free unused GPU memory
                torch.cuda.synchronize()  # Ensure all operations are complete

            return generate_questions

        # Apply Efficient Batch Processing
        set_start_method("spawn", force=True)

        for i, dataset in enumerate(batch_set.data_sets):
            generate_questions = genereate_question_factory(i)
            dataset.map(
                generate_questions,
                batched=True,
                batch_size=self.batch_size,
                with_indices=True,
                num_proc=torch.cuda.device_count(),  # one process per GPU
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(datetime.now(), "All batches processed")


def save_qa_to_parquet(qa_data: pd.DataFrame, file_path: str):
    """save_qa_to_parquet"""
    try:
        qa_data.to_parquet(file_path, index=False)
    except Exception as e:
        print(f"Error saving to parquet: {e}")
