<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Reproducibility Study for Span Type Characteristics

This project aims to reproduce some of the metrics reported by the paper,
*Dissecting Span Identification Tasks with Performance Prediction* (Papay, et
*al, 2020) with a few additions:

- New datasets on Nested NER, particularly GENIA and the Shared BioNLP task.
  The GENIA dataset is sourced from the parsed IOB version in the source repository
  of *A Boundary-aware Neural Model for Nested Named Entity Recognition* (Zheng, et al 2019).
- I will also be adding a few more datasets such as the Healthsea dataset, the
  EBM-NLP corpus, and the ToxicSpans dataset.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install dependencies |
| `riqua` | Parse and analyze the RiQuA (Rich Quotation Analysis) quotation dataset. |
| `conll2000` | Parse the ConLL 2003 English dataset. |
| `conll2003` | Parse and analyze the ConLL 2003 English dataset. |
| `genia` | Parse and analyze the GENIA dataset (corpus from Boundary Aware Nested NER paper) |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `analyze-all` | `riqua` &rarr; `conll2000` &rarr; `conll2003` &rarr; `genia` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->