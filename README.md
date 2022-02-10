# 🪐 [WIP] spacy-span-analyzer

A simple tool to analyze the [Spans](https://spacy.io/api/span) in your
dataset. It's tightly integrated with
[spaCy](https://github.com/explosion/spaCy), so you can easily incorporate it
to existing NLP pipelines. This is also a reproduction of Papay, et al's work on [*Dissecting Span
Identification Tasks with Performance
Prediction*](https://aclanthology.org/2020.emnlp-main.396.pdf) (EMNLP 2020).


## ⏯ Usage

You can use the Span Analyzer as a command-line tool:

```sh
spacy-span-analyzer ./path/to/dataset.spacy
```

Or as an imported library:

```python
import spacy
from spacy.tokens import DocBin
from spacy_span_analyzer import SpanAnalyzer

nlp = spacy.blank("en")  # or any Language model

# Ensure that your dataset is a DocBin
doc_bin = DocBin().from_disk("./path/to/data.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Run SpanAnalyzer and get span characteristics
analyze = SpanAnalyzer(docs)
analyze.frequency  
analyze.length
analyze.span_distinctiveness
analyze.boundary_distinctiveness
```