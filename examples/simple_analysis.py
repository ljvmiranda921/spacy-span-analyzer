import spacy
from spacy.tokens import DocBin
from wasabi import msg

from spacy_span_analyzer import SpanAnalyzer


def main():
    msg.divider("Running simple_analysis.py example")
    nlp = spacy.blank("en")  # or any Language model

    # Ensure that your dataset is a DocBin
    path = "./data/ebm_nlp.spacy"
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    msg.info(f"Loaded {len(docs)} from {path}")

    # Run SpanAnalyzer and get span characteristics
    analyze = SpanAnalyzer(docs)
    msg.text(f"Frequency: {analyze.frequency}")
    msg.text(f"Length: {analyze.length}")
    msg.text(f"Span Distinctiveness: {analyze.span_distinctiveness}")
    msg.text(f"Boundary Distinctiveness: {analyze.boundary_distinctiveness}")


if __name__ == "__main__":
    main()
