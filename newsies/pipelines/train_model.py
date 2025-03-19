"""
newsies.pipelines.train_model

"""

from datetime import datetime
from pathlib import Path
import json
import os
from typing import Any, Dict
import torch
from torch.multiprocessing import set_start_method

from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    BatchEncoding,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import spacy
import pandas as pd


from newsies.chromadb_client import ChromaDBClient
from newsies import targets

# pylint: disable=broad-exception-raised

# Set the torch multiprocessing start method to 'spawn'


def log_memory_usage(stage):
    """log_memory_usage"""
    allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1e9  # Convert to GB
    with open("mem.log", "a", encoding="utf8") as fh:
        fh.writelines(
            f"[{stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n"
        )


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


def save_qa_to_parquet(qa_data, file_path: str):
    """save_qa_to_parquet"""
    df = pd.DataFrame(dict(qa_data))
    df.to_parquet(file_path, index=False)


def load_qa_from_parquet(file_path):
    """load_qa_from_parquet"""
    df = pd.read_parquet(file_path)
    return df.to_dict(orient="records")


def generate_qa_pairs(batch_size=1000, number_of_questions: int = 3):
    """
    generate qa_pairs
    """

    number_of_procs = torch.cuda.device_count()

    news_docs, news_metadata = fetch_news_data()
    print(
        datetime.now(),
        f"start geneating questions for {len(news_docs)} articles\t"
        f"batch_size: {batch_size}\tnumber_of_processes: {number_of_procs}\t"
        f"number_of_questions per prompt: {number_of_questions}",
    )

    nlp = spacy.load("en_core_web_sm")  # load model for detecting

    def extract_named_entities_batch(texts):
        """Batch extract named entities using spaCy's efficient pipe processing"""

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

        entities_list = []

        for doc in nlp.pipe(
            texts, batch_size=32, n_process=4
        ):  # Batch and use multiprocessing
            entities = {ent.text for ent in doc.ents if ent.label_ in relevant_labels}
            entities_list.append(list(entities))  # Convert set to list

        return entities_list

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
            "answer": "",
        }
    )
    # Apply batch NER processing
    df["ne"] = extract_named_entities_batch(df["doc"].tolist())
    del nlp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    df = df.drop(["ne"], axis=1)
    df["answer"] = df.apply(
        lambda row: (
            f"URI:' {row['uri']}'  SECTION: '{row['section']}'  "
            f"HEADLINE: '{row['headline']}'"
        ),
        axis=1,
    )
    dataset = Dataset.from_pandas(df)

    # reclaim memory ahead of
    del df
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    # Define processing function for batch generation
    def generate_questions(batch, indices):
        batch_prompts = batch["prompt"]

        # Tokenize batch (explicit tokenization since we're using .generate())
        # log_memory_usage("Before Tokenization")
        batch_inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        # log_memory_usage("After Tokenization")

        # Compute batch ID from indices
        batch_id: int = indices[0] // batch_size  # First index determines batch
        # log_memory_usage("Before Generation")
        with torch.no_grad():  # disable gradient computation
            # with torch.amp.autocast(device_type=str(device)):  # use mixed precision
            # Generate Questions using Hugging Face Pipeline (Batch Mode)
            batch_questions_encoded = model.generate(
                **batch_inputs,
                max_length=100,
                num_return_sequences=number_of_questions,  # Generate 3 questions per prompt
                do_sample=True,  # Introduce randomness for variation
                temperature=0.7,  # Adjust temperature for diversity
                top_p=0.9,  # Nucleus sampling for more natural responses
            )

        # log_memory_usage("After Generation")
        batch_questions_encoded = batch_questions_encoded.cpu()
        # log_memory_usage("After Moving to CPU")

        batch_questions = tokenizer.batch_decode(
            batch_questions_encoded, skip_special_tokens=True
        )

        # Format questions properly
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
                for ii in range(0, number_of_questions)
            ]
            for i in range(0, len(batch_questions), number_of_questions)
        ]

        batch["question"] = formatted_questions

        # Save Each Batch Immediately to Parquet
        save_qa_to_parquet(
            batch, file_path=f"./training_data/qa_dataset_batch_{batch_id:04d}.parquet"
        )

        # 🔥 Free GPU memory manually
        del batch_inputs, batch_questions_encoded  # Delete tensors
        torch.cuda.empty_cache()  # Free unused GPU memory
        torch.cuda.synchronize()  # Ensure all operations are complete
        # log_memory_usage("After Emptying Cache")

    # Apply Efficient Batch Processing
    set_start_method("spawn", force=True)
    dataset.map(
        generate_questions,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(datetime.now(), "All batches processed")


def load_training_data() -> pd.DataFrame:
    """
    load_training_data
    """
    project_root = os.path.abspath(".")
    df = pd.read_parquet(project_root + "/training_data/")
    return df


def format_dataset(qa_dataset: pd.DataFrame):
    """Ensure tokenizer has a padding token and tokenize dataset."""

    base_model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    def tokenize_sample(sample) -> BatchEncoding:
        """Tokenizes input and output text."""
        question = str(sample["question"]) if sample["question"] is not None else ""
        answer = str(sample["answer"]) if sample["answer"] is not None else ""

        # Tokenize both question and answer with consistent padding length
        tokenized = tokenizer(
            question,
            text_target=answer,  # Proper way to tokenize input + labels
            padding="max_length",  # Force consistent padding
            truncation=True,
            max_length=512,
        )

        return tokenized  # Already contains input_ids, attention_mask, and labels

    # Drop rows where 'question' is NaN or empty
    qa_dataset = qa_dataset.dropna(subset=["question"])
    qa_dataset = qa_dataset[
        qa_dataset["question"].str.strip() != ""
    ]  # Remove empty questions

    dataset = Dataset.from_pandas(qa_dataset)
    tokenized_dataset = dataset.map(
        tokenize_sample,
        remove_columns=[
            "question",
            "answer",
            "uri",
            "section",
            "headline",
            "prompt",
            "doc",
        ],
    )

    return tokenized_dataset.train_test_split(test_size=0.2)


def get_train_and_test_data() -> tuple[Dataset, Dataset]:
    """
    get_train_and_test_data
    """
    # Apply the function
    train_df = load_training_data()
    split_dataset = format_dataset(train_df)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    return (train_dataset, test_dataset)


def train_model() -> tuple[str, pd.DataFrame]:
    """
    train_model
        consumes the latest data in training_data and splits it 80:20 for training and
        test data.
        Trains the LoRA for the mistral model and saves it to the archive
        lora_adaptors/mistral_lora_{YYYmmddHHMM}
        returns a tuple containing the path to the new LoRa and a dataframe of test data
    """
    lora_dir = f"lora_adapters/mistal_lora_{datetime.now().strftime(r'%Y%m%d%H%M')}"
    os.makedirs(f"./{lora_dir}", exist_ok=True)

    train_dataset, test_dataset = get_train_and_test_data()

    # Step 5: Load Model and Apply LoRA Fine-Tuning
    base_model_name = "mistralai/Mistral-7B-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    # Resize token embeddings if a new pad token was added
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

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
    model.save_pretrained(lora_dir)
    with open("./lora_adaptors.txt", "a", encoding="utf8") as fh:
        fh.write(f"{lora_dir}\n")

    # clear the cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (lora_dir, test_dataset)


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


def test_lora(lora_dir: str, test_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Loads a base Mistral-7B-v0.3 model and applies a LoRA adapter,
    then evaluates it using the provided test dataset.

    Args:
        lora_dir (str): Path to the directory containing the saved LoRA adapter.
        test_data (pd.DataFrame): Test dataset containing input prompts and expected outputs.

    Returns:
        Dict[str, Any]: Dictionary with model predictions and validation metrics.
    """

    # Load Base Model & Tokenizer
    base_model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Load LoRA Adapter
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()  # Set to evaluation mode

    # Convert test_data DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(test_data)

    def generate_output(sample):
        """Generates text using the model given a prompt."""
        input_text = sample["question"] if "question" in sample else sample["prompt"]
        input_ids = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=512)

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"generated_answer": generated_text}

    # Generate predictions
    results = dataset.map(generate_output)

    # Compare with expected answers (if available)
    predictions = results["generated_answer"]
    expected = results["answer"] if "answer" in results.column_names else None

    # Construct validation dictionary
    output_dict = {
        "predictions": predictions,
        "expected": expected if expected else "N/A",
        "success": all(
            isinstance(pred, str) and len(pred) > 0 for pred in predictions
        ),  # Basic check for valid outputs
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dict


def get_latest_lora_adapter():
    """
    get_latest_lora_adapter
    """
    with open("./lora_adaptors.txt", "rb") as fh:
        try:
            fh.seek(-2, os.SEEK_END)
            while fh.read(1) != b"\n":
                fh.seek(-2, os.SEEK_CUR)
            last_line = fh.readline().decode(encoding="utf8")
        except OSError:
            last_line = fh.readline().decode(encoding="utf8")
    return last_line


def get_latest_test_data() -> pd.DataFrame:
    """
    get_latest_test_data
    """
    dirs = os.listdir("./test_data")
    dirs.sort()
    latest_dir = dirs[-1]
    return pd.read_parquet(f"./test_data/{latest_dir}/test_data.parquet")
