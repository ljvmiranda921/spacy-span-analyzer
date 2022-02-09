import math
from collections import Counter
from typing import Dict, List, Union, Set, Tuple

from scipy.stats.mstats import gmean
from spacy.tokens import Doc, Span, SpanGroup, Token


class SpanAnalyzer:
    # TODO: do this for all spans_key
    def __init__(self, docs: List[Doc]):
        self.docs = docs

    @property
    def frequency(self) -> Dict[str, Counter]:
        """Number of spans for a span type in the dataset's training corpus."""
        frequency = {}
        for doc in self.docs:
            for spans_key in list(doc.spans.keys()):
                if spans_key not in frequency:
                    frequency[spans_key] = Counter()
                for span in doc.spans[spans_key]:
                    if span.label_ is None:
                        continue
                    else:
                        frequency[spans_key][span.label_] += 1
        return frequency

    @property
    def length(self) -> Dict[str, float]:
        """Geometric mean of the spans' lengths in tokens."""
        _length = {}
        for doc in self.docs:
            for spans_key in list(doc.spans.keys()):
                if spans_key not in _length:
                    _length[spans_key] = []
                for span in doc.spans[spans_key]:
                    _length[spans_key].append(len(span))

        length = {spans_key: gmean(lengths) for spans_key, lengths in _length.items()}
        return length

    @property
    def span_distinctiveness(self) -> Dict[str, float]:
        """Distinctiveness of the span compared to the corpus."""
        p_corpus = self._get_unigram_distribution(self.docs, normalize=True)
        all_spans_key = self._get_all_keys()

        p_spans = {}
        for key in all_spans_key:
            spans = self._get_all_spans_in_key(key)
            p_span = self._get_unigram_distribution(spans, normalize=True)
            p_spans[key] = p_span

        # Compute value for each spans_key
        span_distincts = {
            key: self._get_kl_divergence(p_span, p_corpus)
            for key, p_span in p_spans.items()
        }

        return span_distincts

    @property
    def boundary_distinctiveness(self):
        """Distinctiveness of the boundaries compared to the corpus."""
        # TODO
        pass

    def _get_all_keys(self) -> Set[str]:
        """Get all spans_key in the corpus"""
        return set([key for doc in self.docs for key in list(doc.spans.keys())])

    def _get_all_spans_in_key(self, spans_key: str) -> List[Span]:
        """Get all spans given a specified spans_key"""
        return [span for doc in self.docs for span in doc.spans[spans_key]]

    def _get_all_boundaries_in_key(
        self, spans_key: str
    ) -> Tuple[List[Token], List[Token]]:
        """Get the boundary tokens for all spans in a spans_key"""
        starts: List[Token] = []
        ends: List[Token] = []
        for doc in self.docs:
            for span in doc.spans[spans_key]:
                span_bound_start_idx = span.start - 1
                if span_bound_start_idx >= 0:
                    starts.append(doc[span_bound_start_idx])

                span_bound_end_idx = span.end + 1
                if span_bound_end_idx <= len(doc):
                    ends.append(doc[span_bound_end_idx])

        return (starts, ends)

    def _get_unigram_distribution(
        self, texts: Union[List[Doc], List[SpanGroup]], normalize: bool = True
    ) -> Counter:
        """Get each word's sample frequency given a list of text"""
        word_counts = Counter()
        for text in texts:
            for token in text:
                word_counts[self._normalize_text(token.text)] += 1

        if normalize:
            total = sum(word_counts.values(), 0.0)
            word_counts = Counter({k: v / total for k, v in word_counts.items()})

        return word_counts

    def _get_kl_divergence(self, p: Counter, q: Counter) -> float:
        """Compute the Kullback-Leibler divergence

        The parameters p and q are taken as unigram word distributions
        """
        total = 0
        for word, p_word in p.items():
            total += p_word * math.log(p_word / q[word])
        return total

    def _normalize_text(text: str) -> str:
        # The source code from the paper converts all digits into a single
        # value, '0'. Perhaps this is a way to "normalize" all numerical value
        # I'm not sure if this is generalizable so I'll comment it out for the meantime.

        # pattern = re.compile("[0-9]")
        # text = re.sub(pattern, "0", text)
        text = text.lower().replace("``", '"').replace("''", '"')
        return text
