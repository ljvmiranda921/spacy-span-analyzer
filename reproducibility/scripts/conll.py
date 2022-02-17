import zipfile
from enum import Enum
from typing import Literal
from pathlib import Path
from spacy import load

import spacy
import typer
from datasets import load_dataset
from wasabi import msg
from spacy.tokens import DocBin, Doc, SpanGroup

from .constants import ASSETS_PATH, CORPUS_PATH


class ConLLDataset(str, Enum):
    conll2000 = "conll2000"
    conll2003 = "conll2003"


NER_COL_MAP = {"conll2000": "chunk_tags", "conll2003": "ner_tags"}


def main(dataset: ConLLDataset = ConLLDataset.conll2000):
    """Download and parse ConLL datasets"""
    msg.info("Getting dataset from Huggingface hub (train split only)")
    hub_dataset = load_dataset(dataset.value, split="train")
    nlp = spacy.blank("en")

    ner_tag_map = hub_dataset.features[NER_COL_MAP[dataset.value]].feature.names

    docs = []
    for tokens, tags in zip(
        hub_dataset["tokens"], hub_dataset[NER_COL_MAP[dataset.value]]
    ):
        ner_tags = [ner_tag_map[tag] for tag in tags]
        doc = Doc(nlp.vocab, tokens, ents=ner_tags)
        docs.append(doc)

    # Transfer doc.ents to doc.spans
    for doc in docs:
        group = SpanGroup(doc, name="sc", spans=list(doc.ents))
        doc.spans["sc"] = group

    # Save to doc_bin
    msg.info("Saving into DocBin")
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(CORPUS_PATH / f"{dataset.value}.spacy")
    msg.good(f"Saved to {CORPUS_PATH}")


if __name__ == "__main__":
    typer.run(main)
