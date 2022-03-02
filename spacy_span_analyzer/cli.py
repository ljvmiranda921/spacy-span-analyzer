import json
from inspect import cleandoc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
import typer
from spacy.tokens import Doc, DocBin
from wasabi import msg

from .analyzer import SpanAnalyzer, weighted_average


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to .spacy file"
    ),
    output: Path = typer.Option(
        None, exists=False, dir_okay=False, help="Path to save metrics as JSON file"
    ),
    spacy_model: Optional[str] = typer.Option(
        None, help="Loadable spaCy pipeline (uses spacy.blank('en') if not provided)"
    ),
    window_size: int = typer.Option(
        1, help="Window size to consider for the span boundaries"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show descriptions for each span property"
    ),
):
    nlp = spacy.load(spacy_model) if spacy_model else spacy.blank("en")

    # Read DocBin file
    doc_bin = DocBin().from_disk(input_path)
    docs: List[Doc] = list(doc_bin.get_docs(nlp.vocab))
    msg.info(f"Loaded {len(docs)} from {input_path}")

    # Perform span analysis
    analyzer = SpanAnalyzer(docs, window_size=window_size)
    msg.text(f"Spans Keys: {list(analyzer.keys)}")

    # Store in a variable because we will reuse this
    frequencies = analyzer.frequency

    msg_template(
        data=frequencies,
        divider="Span Type Frequency",
        header=("Span Type", "Frequency"),
        doc=SpanAnalyzer.frequency.__doc__,
        verbose=verbose,
    )

    msg_template(
        data=analyzer.length,
        divider="Span Type Length",
        header=("Span Type", "Length"),
        doc=SpanAnalyzer.length.__doc__,
        verbose=verbose,
        frequencies=frequencies,
    )

    msg_template(
        data=analyzer.span_distinctiveness,
        divider="Span Distinctiveness",
        header=("Span Key", "Span Distinctiveness"),
        doc=SpanAnalyzer.span_distinctiveness.__doc__,
        verbose=verbose,
        frequencies=frequencies,
    )

    msg_template(
        data=analyzer.boundary_distinctiveness,
        divider=f"Span Boundary Distinctiveness (window={analyzer.window_size})",
        header=("Span Key", "Boundary Distinctiveness"),
        doc=SpanAnalyzer.boundary_distinctiveness.__doc__,
        verbose=verbose,
        frequencies=frequencies,
    )

    if output:
        output_dict = {
            "metrics": {
                "frequencies": dict(analyzer.frequency),
                "length": analyzer.length,
                "span_distinctiveness": analyzer.span_distinctiveness,
                "boundary_distinctiveness": analyzer.boundary_distinctiveness,
            },
            "config": {"window_size": analyzer.window_size},
        }
        with output.open("w") as output_file:
            json.dump(output_dict, output_file, indent=4)
        msg.good(f"Saved to {output}!")


def msg_template(
    data: Dict[str, Any],
    divider: str,
    header: Tuple,
    doc: str,
    verbose: bool,
    frequencies: Optional[Dict[str, float]] = None,
):
    msg.divider(divider)
    if verbose:
        msg.text(cleandoc(doc))
    for spans_key, values in data.items():
        msg.text(f"In spans key: {spans_key}")
        msg.table(values, header=header, divider=True)

    if frequencies:
        # Compute weighted average
        w_avg = weighted_average(data, frequencies)
        msg.text("Weighted Average (by frequency)")
        msg.table(w_avg, header=("Spans Key", "Average"), divider=True)


if __name__ == "__main__":
    app()
