"""
data_framer.py
"""

from typing import Any
import pandas as pd
from datasets import Dataset
import torch

from newsies.llm.batch_set import BatchSet


class DataFramer:
    """
    DataFramer class is used to convert batches of data into a pandas DataFrame.
    """

    def visit(self, batch_set: Any):
        """visit method to accept a batch set."""
        match batch_set:
            case BatchSet():
                batch_set.accept(self)
            case _:
                raise TypeError(
                    f"DataFramer only accepts BatchSet, got {type(batch_set)}"
                )

    def visit_batch_set(self, batch_set: BatchSet):
        """
        Visit the BatchSet class and convert batches to pandas DataFrames.
        """
        for (pubdate, batches), news_metadata, articles in batch_set:
            for idx, batch in enumerate(batches):
                df = pd.DataFrame(
                    {
                        "pubdate": pubdate,
                        "batch": ",".join(batch),
                        "doc": [a.story for a in articles[idx]],
                        "uri": [a.path() for a in articles[idx]],
                        "section0": [
                            meta["section0"] or "front-page"
                            for meta in news_metadata[idx]
                        ],
                        "headline0": [meta["headline0"] for meta in news_metadata[idx]],
                        "section1": [
                            meta.get("section1") for meta in news_metadata[idx]
                        ],
                        "headline1": [
                            meta.get("headline1") for meta in news_metadata[idx]
                        ],
                        "section2": [
                            meta.get("section2") for meta in news_metadata[idx]
                        ],
                        "headline2": [
                            meta.get("headline2") for meta in news_metadata[idx]
                        ],
                        "answer": "",
                        "question": "",
                        "train_question": "",
                        "prompt": "",
                        "ne": [meta["ner_terms"] for meta in news_metadata[idx]],
                    }
                )

                df1 = df[(df["section1"] != "N/A") & (df["headline1"] != "N/A")]
                df1 = df1.drop(["section0", "headline0"], axis=1)
                df1 = df1.rename(
                    columns={"section1": "section", "headline1": "headline"}
                )

                df2 = df1[(df1["section2"] != "N/A") & (df1["headline2"] != "N/A")]
                df2 = df2.drop(["section", "headline"], axis=1)
                df2 = df2.rename(
                    columns={"section2": "section", "headline2": "headline"}
                )

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
                batch_set.data_sets.append(dataset)

        # reclaim memory ahead of
        del df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
