# spacy-span-analyzer

A simple tool to analyze the [Spans](https://spacy.io/api/span) in your
dataset. It's tightly integrated with
[spaCy](https://github.com/explosion/spaCy), so you can easily incorporate it
to existing NLP pipelines. This is also a reproduction of Papay, et al's work on [*Dissecting Span
Identification Tasks with Performance
Prediction*](https://aclanthology.org/2020.emnlp-main.396.pdf) (EMNLP 2020).

## ⏳ Install

Using
[pip](https://packaging.python.org/en/latest/tutorials/installing-packages/):

```sh
pip install spacy-span-analyzer
```

Directly from source (I highly recommend running this within a [virtual
environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)):

```sh
git clone git@github.com:ljvmiranda921/spacy-span-analyzer.git
cd spacy-span-analyzer
pip install .
```

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

Inputs are expected to be a list of spaCy [Docs](https://spacy.io/api/doc) or a [DocBin](https://spacy.io/api/docbin) (if you're using
the command-line tool).

### Working with Spans

In spaCy, you'd want to store your Spans in the
[`doc.spans`](https://spacy.io/api/doc#spans) property, under a particular
`spans_key` (`sc` by default). Unlike the
[`doc.ents`](https://spacy.io/api/doc#ents) property, `doc.spans` allows
overlapping entities. This is useful especially for downstream tasks like [Span
Categorization](https://spacy.io/api/spancategorizer). 

A common way to do this is to use
[`char_span`](https://spacy.io/api/doc#char_span) to define a slice from your
Doc:

```python
doc = nlp(text)
spans = []
from annotation in annotations:
    span = doc.char_span(
        annotation["start"],
        annotation["end"],
        annotation["label"],
    )
    spans.append(span)

# Put all spans under a spans_key
doc.spans["sc"] = spans
```

You can also achieve the same thing by using
[`set_ents`](https://spacy.io/api/doc#set_ents) or by creating a 
[SpanGroup](https://spacy.io/api/spangroup).