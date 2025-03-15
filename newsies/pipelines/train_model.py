"""
newsies.pipelines.train_model

"""

from datetime import datetime
from pathlib import Path
import json
import os

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from tqdm.notebook import tqdm
import spacy
from datasets import Dataset

import pandas as pd


from newsies.chromadb_client import ChromaDBClient
from newsies import targets


# Step 1: Connect to ChromaDB and Retrieve Data
def fetch_news_data(collection_name: str = "ap_news_2025-03-12"):
    """fetch_news_data"""
    client = ChromaDBClient()  # Update path
    client.collection_name = collection_name
    print(f"collection name: {client.collection.name}")
    collection = client.collection
    n = collection.count()
    print(f"there are {n} stories in the collection")
    results = collection.get(where={"target": {"$eq": targets.DOCUMENT}}, limit=n)
    return results["documents"], results["metadatas"]


# Step 2: Generate Question-Answer Pairs using an LLM
def extract_named_entities(text):
    """extract_named_entities"""
    # Load spaCy model for Named Entity Recognition (NER)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = list(
        set(ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"})
    )
    return entities


def save_debug_output(prompt, results):
    """
    save_debug_output
    """
    debug_data = []
    debug_data.append({"qty_prompts": len(prompt), "qty_results": len(results)})
    for q, a in zip(prompt, results):
        debug_data.append({"prompt": q, "question": a})
    with open("debug_missing_questions.jsonl", "w", encoding="utf8") as fh:
        fh.write(json.dumps(debug_data, indent=4))


def save_qa_to_parquet(qa_data, file_path):
    """save_qa_to_parquet"""
    df = pd.DataFrame(qa_data)
    df.to_parquet(file_path, index=False)


def load_qa_from_parquet(file_path):
    """load_qa_from_parquet"""
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")


# Modify the QA generation function
def generate_qa_pairs(
    news_docs, news_metadata, batch_size=1000, entity_batch_size=1000
):
    """generate_qa_pairs"""

    qa_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0 if torch.cuda.is_available() else -1,
    )
    qa_data = []
    debug_data = []  # Store problematic responses
    total_batches = (len(news_docs) + batch_size - 1) // batch_size

    print(datetime.now(), "start processing generated questions for training")
    for batch_start in tqdm(
        range(0, len(news_docs), batch_size),
        desc=f"Processing {total_batches} Batches {datetime.now()}",
        position=0,
    ):
        batch_docs = news_docs[batch_start : batch_start + batch_size]
        batch_meta = news_metadata[batch_start : batch_start + batch_size]

        question_prompts = []
        entity_prompts = []
        entity_mapping = (
            []
        )  # Keep track of which entity question belongs to which article

        for doc, meta in zip(batch_docs, batch_meta):
            context = (
                f"'section': {meta['section0'] or 'front-page'}\t"
                f"'headline':{meta['headline0']}\n"
            )
            if meta["section1"] != "N/A":
                context += (
                    f" [section]: {meta['section1']}\t'headline': {meta['headline1']}\n"
                )
            if meta["section2"] != "N/A":
                context += (
                    f" 'section': {meta['section2']}\t'headline': {meta['headline2']}\n"
                )
            context += f"'URI': {meta['uri']}\n"
            context += f"'article': {doc}"

            # Generate 3 diverse questions about the article
            question_prompts.append(
                "Generate 3 different questions that a reader might ask about the "
                "following news article. Focus on specific facts, key events, or important "
                "themes in the article. The questions should be clear, meaningful, and "
                "relevant to the article's details. The questions should avoid generic inquiries. "
                "Ensure the question cannot be answered without reading the article.\n"
                f"'news article': {doc}"
            )

            # Collect entities in the story
            # Exception(s):  AP
            entities = [e for e in extract_named_entities(doc) if e != "AP"]

            # Generate questions for each entity separately
            for entity in entities:
                entity_prompts.append(
                    f"Generate a question about '{entity}' that requires knowledge of the "
                    "following news article. Focus on specific facts, key events, or important "
                    "themes in the article. The questions should be clear, meaningful, and "
                    "relevant to the article's details. The questions should avoid generic "
                    "inquiries. Ensure the question cannot be answered without reading the "
                    "article.\n"
                    f"'news article': {doc}"
                )
                entity_mapping.append(
                    (doc, meta)
                )  # Track which article each entity belongs to

        # print(datetime.now(), f"Processing batch {batch_start // batch_size + 1}/{total_batches}")

        article_questions = qa_generator(
            question_prompts, max_length=50, truncation=True
        )
        # Store results for articles
        for (doc, meta), prompt, article_question_output in zip(
            zip(batch_docs, batch_meta), question_prompts, article_questions
        ):
            if isinstance(article_question_output["generated_text"], str):
                questions = [
                    q
                    for q in article_question_output["generated_text"].split("\n")
                    if q != ""
                ]
            if isinstance(article_question_output["generated_text"], list):
                questions = article_question_output["generated_text"]
                for q in questions:
                    if "\n" in q:
                        questions.append(*[qq for qq in q.split()])
            for question in questions:
                answer = f"'URI'  {meta['uri']}\n"
                for i in range(3):
                    if meta[f"section{i}"] != "N/A":
                        answer += (
                            f"'section' {meta[f"section{i}"] or "front-page"}\t"
                            f"'headline' {meta[f'headline{i}']}\n"
                        )

                qa_data.append(
                    {
                        "question": (
                            "for the next question, return the 'section', "
                            "the 'headline', and the 'URI'\n"
                            f"question: '{question}'"
                        ),
                        "context": context,
                        "answer": answer,
                    }
                )

        # Process entity-related questions in batches
        total_entity_batches = (
            len(entity_prompts) + batch_size - 1
        ) // entity_batch_size
        for entity_batch_start in tqdm(
            range(0, len(entity_prompts), entity_batch_size),
            desc=f"Processing {total_entity_batches} Entity Batches {datetime.now()}",
            position=1,
            leave=False,
        ):
            entity_batch = entity_prompts[
                entity_batch_start : entity_batch_start + entity_batch_size
            ]

            entity_results = qa_generator(entity_batch, max_length=50, truncation=True)

            # Log responses if no valid questions are found
            if entity_results is None or len(entity_results) < len(entity_batch):
                print(
                    f"Mismatch detected! entity_batch: {len(entity_batch)}, "
                    f"entity_results: {len(entity_results)}"
                )
                print(
                    f"entity results type: {type(entity_results)}\t"
                    f"entity_batch type: {type(entity_batch)}"
                )
                debug_data.append(
                    {"context": entity_batch, "raw_output": entity_results}
                )
                save_debug_output(entity_batch, entity_results)
                raise Exception("ERROR: please review debug_missing_questions.jsonl")

            for (doc, meta), prompt, entity_question_output in zip(
                entity_mapping[
                    entity_batch_start : entity_batch_start + entity_batch_size
                ],
                entity_prompts,
                entity_results,
            ):
                if isinstance(entity_question_output["generated_text"], str):
                    questions = [
                        q
                        for q in entity_question_output["generated_text"].split("\n")
                        if q != ""
                    ]
                if isinstance(entity_question_output["generated_text"], list):
                    questions = entity_question_output["generated_text"]
                    for q in questions:
                        if "\n" in q:
                            questions.append(*[qq for qq in q.split()])
                for question in questions:
                    answer = f"'URI'  {meta['uri']}\n"
                    for i in range(3):
                        if meta[f"section{i}"] != "N/A":
                            answer += (
                                f"'section' {meta[f"section{i}"] or "front-page"}\t"
                                f"'headline' {meta[f'headline{i}']}\n"
                            )
                    qa_data.append(
                        {
                            "question": (
                                "for the next question, return the 'section', "
                                "the 'headline', and the 'URI'\n"
                                f"question: '{question}'"
                            ),
                            "context": context,
                            "answer": answer,
                        }
                    )

        # Save batch results and clear memory
        batch_file = (
            f"training_data/qa_dataset_batch_{batch_start // batch_size + 1}.parquet"
        )
        save_qa_to_parquet(qa_data, batch_file)
        qa_data.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(datetime.now(), "All batches processed")
    del qa_generator
    torch.cuda.empty_cache()


def get_training_data() -> pd.DataFrame:
    """
    get_training_data
    """
    project_root = os.path.abspath(".")
    train_df = pd.read_parquet(project_root + "/notebooks/training_data")
    return train_df


def format_dataset(qa_dataset):
    """Ensure tokenizer has a padding token and tokenize dataset."""

    base_model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    def tokenize_sample(sample):
        """Tokenizes input and output text."""
        question = str(sample["question"]) if sample["question"] is not None else ""
        answer = str(sample["answer"]) if sample["answer"] is not None else ""

        inputs = tokenizer(question, padding=True, truncation=True, max_length=512)
        outputs = tokenizer(answer, padding=True, truncation=True, max_length=512)

        inputs["labels"] = outputs["input_ids"]  # Assign tokenized answers as labels
        return inputs

    # Drop rows where 'question' is NaN or empty
    qa_dataset = qa_dataset.dropna(subset=["question"])
    qa_dataset = qa_dataset[
        qa_dataset["question"].str.strip() != ""
    ]  # Remove empty questions

    dataset = Dataset.from_pandas(qa_dataset)
    tokenized_dataset = dataset.map(
        tokenize_sample, remove_columns=["question", "answer"]
    )

    return tokenized_dataset.train_test_split(test_size=0.2)


def get_train_and_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    get_train_and_test_data
    """
    # Apply the function
    train_df = get_training_data()
    split_dataset = format_dataset(train_df)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    return (train_dataset, test_dataset)


def train_model():
    """
    train_model
    """
    train_dataset, test_dataset = get_train_and_test_data()

    # Step 5: Load Model and Apply LoRA Fine-Tuning
    base_model_name = "mistralai/Mistral-7B-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./news_finetune_model",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        remove_unused_columns=False,  # Ensure model gets correct inputs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()


def download_mistral():
    """
    download_mistral
    this should be used only in installation
    """

    mistral_models_path = Path.home().joinpath("mistral_models", "7B-v0.3")
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    #
    snapshot_download(
        repo_id="mistralai/Mistral-7B-v0.3",
        allow_patterns=[
            "params.json",
            "consolidated.safetensors",
            "tokenizer.model.v3",
        ],
        local_dir=mistral_models_path,
    )
