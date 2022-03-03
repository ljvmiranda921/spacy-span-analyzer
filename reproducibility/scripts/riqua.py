import tarfile
from pathlib import Path
from typing import List, Set, Union

import requests
import spacy
import typer
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from wasabi import msg

from .constants import ASSETS_PATH, CORPUS_PATH

URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/riqua/riqua.tar.gz"
FILENAME = "riqua.tar.gz"
ANNOTATIONS = ASSETS_PATH / "riqua" / "merged"


def download_riqua(url: str) -> Path:
    save_path = ASSETS_PATH / FILENAME
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-Length", 0))

    with open(save_path, "wb") as file:
        pbar = tqdm(unit="B", total=total)
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                pbar.update(len(chunk))
                file.write(chunk)
    return save_path


def _convert_to_doc(
    file: str, nlp: spacy.language.Language, spans_key: str = "sc"
) -> Doc:
    """Convert BRAT into spaCy

    Entities are saved into doc.spans, not doc.ents for the purposes
    of this reproducibility study.
    """
    text_file = ANNOTATIONS / f"{file}.txt"
    annot_file = ANNOTATIONS / f"{file}.ann"

    with open(text_file, "r") as f:
        text_str = f.read()

    annotations = []
    with open(annot_file, "r") as f:
        for line in f:
            entry = line.split()
            if entry[0].startswith("T"):
                data = {
                    "label": entry[1],
                    "start": int(entry[2]),
                    "end": int(entry[3]),
                    "text": " ".join(entry[4:]),
                }
                annotations.append(data)

    # Customize the tokenizer so it splits on double-hyphen em-dashes
    suffixes = nlp.Defaults.suffixes + ["--"]
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    doc = nlp(text_str)
    spans = []

    for annot in annotations:
        # FIXME: Maybe there's better alignment rules here
        # I'll just set the mode to expand just to see.
        span = doc.char_span(
            annot["start"],
            annot["end"],
            annot["label"],
            alignment_mode="expand",
        )
        if span is None:
            msg.warn(f"Found an empty span in {annot_file}: {annot}")
        spans.append(span)

    doc.spans[spans_key] = spans
    return doc


def parse_riqua(files: Union[List[str], Set[str]]) -> List[Doc]:
    docs = []
    nlp = spacy.blank("en")
    for file in tqdm(files):
        doc = _convert_to_doc(file, nlp)
        docs.append(doc)
    return docs


def main(
    url: str = typer.Option(default=URL, show_default=True),
    skip_download: bool = typer.Option(False, "--skip-download"),
):
    msg.info(f"Downloading files and extracting them")
    filepath = download_riqua(url) if not skip_download else ASSETS_PATH / FILENAME
    with tarfile.open(filepath) as f:
        for member in tqdm(f.getmembers(), total=len(f.getmembers())):
            f.extract(member, path=ASSETS_PATH)

    msg.info(f"Converting annotations into spaCy docs")
    files = set([f.stem for f in ANNOTATIONS.glob("**/*") if f.is_file()])
    if files:
        docs = parse_riqua(files)
    else:
        msg.fail(f"No files found in {ANNOTATIONS}")

    msg.info(f"Saving into DocBin")
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(CORPUS_PATH / "riqua.spacy")
    msg.good(f"Saved to {CORPUS_PATH}")


if __name__ == "__main__":
    typer.run(main)
