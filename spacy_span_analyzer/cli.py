from inspect import cleandoc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
import typer
from spacy.tokens import Doc, DocBin
from wasabi import msg

from .analyzer import SpanAnalyzer


def main(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to .spacy file"
    ),
    spacy_model: Optional[str] = typer.Option(
        None, help="Loadable spaCy pipeline (uses spacy.blank('en') if not provided)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show descriptions for each span property"
    ),
):
    nlp = spacy.load(spacy_model) if spacy_model else spacy.blank("en")

    doc_bin = DocBin().from_disk(input_path)
    docs: List[Doc] = list(doc_bin.get_docs(nlp.vocab))
    msg.info(f"Loaded {len(docs)} from {input_path}")

    # Perform span analysis
    analyzer = SpanAnalyzer(docs)
    msg.text(f"Spans Keys: {list(analyzer.keys)}")

    msg.divider("Span Type Frequency")
    if verbose:
        msg.text(cleandoc(SpanAnalyzer.frequency.__doc__))
    for span_key, counts in analyzer.frequency.items():
        msg.text(f"In span_key: {span_key}")
        msg.table(counts, header=("Span Type", "Frequency"), divider=True)

    msg_template(
        analyzer.length,
        "Span Length",
        ("Span Key", "Length"),
        SpanAnalyzer.length.__doc__,
        verbose=verbose,
    )

    msg_template(
        analyzer.span_distinctiveness,
        "Span Distinctiveness",
        ("Span Key", "Span Distinctiveness"),
        SpanAnalyzer.span_distinctiveness.__doc__,
        verbose=verbose,
    )

    msg_template(
        analyzer.boundary_distinctiveness,
        "Span Boundary Distinctiveness",
        ("Span Key", "Boundary Distinctiveness"),
        SpanAnalyzer.boundary_distinctiveness.__doc__,
        verbose=verbose,
    )


def msg_template(
    data: Dict[str, Any], divider: str, header: Tuple, doc: str, verbose: bool
):
    msg.divider(divider)
    if verbose:
        msg.text(cleandoc(doc))
    msg.table(data, header=header, divider=True)


if __name__ == "__main__":
    typer.run(main)
