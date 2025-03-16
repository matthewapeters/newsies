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
from tqdm import tqdm
import spacy
import pandas as pd


from newsies.chromadb_client import ChromaDBClient
from newsies import targets

# pylint: disable=broad-exception-raised


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

    # Label	Description
    #
    # PERSON	Named persons (e.g., Albert Einstein)
    # ORG	Organizations, companies, agencies (e.g., Apple, CIA)
    # GPE	Countries, cities, states (e.g., France, New York)
    # LOC	Non-GPE locations, like mountains, rivers (e.g., Everest, Amazon River)
    # FAC	Facilities (e.g., Empire State Building, Golden Gate Bridge)
    # NORP	Nationalities, religious or political groups (e.g., American, Buddhist, Republican)
    # DATE	Absolute/relative dates or periods (e.g., June 20, 1990, next week)
    # TIME	Times of the day (e.g., 2:30 PM, midnight)
    # MONEY	Monetary values (e.g., $10, 500 euros)
    # PERCENT	Percentage expressions (e.g., 10% increase)
    # CARDINAL	Numbers that do not fall into other categories (e.g., one, 200 million)
    # ORDINAL	Rankings (e.g., first, 2nd place)
    # QUANTITY	Measurements (e.g., 3kg, 5 miles)
    # PRODUCT	Objects, vehicles, food, etc. (e.g., iPhone, Boeing 747)
    # EVENT	Named events (e.g., World War II, Super Bowl)
    # LAW	Named laws, treaties, regulations (e.g., First Amendment, GDPR)
    # WORK_OF_ART	Titles of books, songs, movies (e.g., The Great Gatsby, Star Wars)
    # LANGUAGE	Any named language (e.g., French, Python)

    relevant_labels = {
        "EVENT",
        "FAC",
        "GPE",
        "LAW",
        "LOC",
        "NORP",
        "ORG",
        "PERSON",
        "PRODUCT",
        "WORK_OF_ART",
    }

    # Load spaCy model for Named Entity Recognition (NER)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = list(set(ent.text for ent in doc.ents if ent.label_ in relevant_labels))
    return entities


def extract_named_entities_batch(texts):
    """Batch extract named entities using spaCy's efficient pipe processing"""
    entities_list = []
    nlp = spacy.load("en_core_web_sm")

    for doc in nlp.pipe(
        texts, batch_size=32, n_process=4
    ):  # Batch and use multiprocessing
        entities = {
            ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}
        }
        entities_list.append(list(entities))  # Convert set to list

    return entities_list


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


def generate_qa_pairs(batch_size=1000, number_of_questions: int = 3):
    """
    generate qa_pairs2
    """

    news_docs, news_metadata = fetch_news_data()
    df = pd.DataFrame(
        {
            "doc": news_docs,
            "uri": [meta["uri"] for meta in news_metadata],
            "section0": [meta["section0"] or "front-page" for meta in news_metadata],
            "headline0": [meta["headline0"] for meta in news_metadata],
            "section1": [meta["section1"] for meta in news_metadata],
            "headline1": [meta["headline1"] for meta in news_metadata],
            "section2": [meta["section2"] for meta in news_metadata],
            "headline2": [meta["headline2"] for meta in news_metadata],
        }
    )

    # Apply batch NER processing
    df["ne"] = extract_named_entities_batch(df["doc"].tolist())

    df1 = df[(df["section1"] != "N/A") & (df["headline1"] != "N/A")]
    df1 = df1.drop(["section0", "headline0"], axis=1)
    df1 = df1.rename(columns={"section1": "section", "headline1": "headline"})

    df2 = df1[(df1["section2"] != "N/A") & (df1["headline2"] != "N/A")]
    df2 = df2.drop(["section", "headline"], axis=1)
    df2 = df2.rename(columns={"section2": "section", "headline2": "headline"})

    df = df.drop(["section1", "headline1", "section2", "headline2"], axis=1)
    df = df.rename(columns={"section0": "section", "headline0": "headline"})
    df1 = df1.drop(["section2", "headline2"], axis=1)
    df = pd.concat([df, df1, df2], ignore_index=True, sort=False)

    dfnoents = df[df["ne"].apply(len) == 0].copy()

    dfne = df[df["ne"].apply(len) > 0].copy()
    dfne = dfne.explode("ne")

    # add prompt
    dfnoents["prompt"] = dfnoents.apply(
        lambda row: (
            "Generate 3 different questions that a reader might ask about the "
            "following news article. Focus on specific facts, key events, or "
            "important themes in the article. The questions should be clear, "
            "meaningful, and relevant to the article's details. The questions "
            "should avoid generic inquiries. Ensure the question cannot be "
            "answered without reading the article.\n"
            f"news article: {row['doc']}"
        ),
        axis=1,
    )

    # add prompt
    dfne["prompt"] = dfne.apply(
        lambda row: (
            f"Generate a question about '{row['ne']}' that requires knowledge "
            "of the following news article. Focus on specific facts, key "
            "events, or important themes in the article. The questions should be "
            "clear, meaningful, and relevant to the article's details. The questions "
            "should avoid generic inquiries. Ensure the question cannot be "
            "answered without reading the article.\n"
            f"news article: {row['doc']}"
        ),
        axis=1,
    )

    df = pd.concat([dfnoents, dfne], ignore_index=True, sort=False)
    df = df.drop(["ne", "doc"], axis=1)
    df["batch"] = df.index // batch_size
    prompt_count = len(df)
    batched_data = df.groupby("batch")
    # reclaim memory ahead of
    del df

    qa_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0 if torch.cuda.is_available() else -1,
    )

    questions = []
    # Run qa_generator on question_prompt_ds - this should be batched to batch_size
    for batch, batch_data in tqdm(
        batched_data,
        desc=(
            f"Generate {number_of_questions} questions for "
            f"{prompt_count} prompts in batches of {batch_size}"
        ),
        position=0,
    ):

        # Process the batch separately by extracting 'prompt' field
        batch_prompts = batch_data["prompt"].tolist()

        # Call qa_generate with the batch of prompts
        batch_questions = qa_generator(
            batch_prompts,
            max_length=100,
            truncation=True,
            num_return_sequences=number_of_questions,
            do_sample=True,  # Introduce randomness for variation
            temperature=0.7,  # Adjust temperature for diversity
            top_p=0.9,  # Nucleus sampling for more natural responses)
        )
        batch_questions = [
            [
                (
                    "for the next question, return the 'section', "
                    "the 'headline', and the 'URI'\n"
                    f"question: '{v}'"
                )
                for d in qs
                for v in d.values()
            ]
            for qs in batch_questions
        ]

        batch_data["question"] = batch_questions

        batch_file = f"training_data/qa_dataset_batch_{batch}.parquet"
        save_qa_to_parquet(batch_data, batch_file)

        questions.extend(batch_questions)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(datetime.now(), "All batches processed")


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
