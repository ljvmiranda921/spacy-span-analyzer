import math
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from scipy.stats.mstats import gmean
from spacy.tokens import Doc, Span, SpanGroup, Token


FrequencyDist = Dict[str, float]
SpanBoundaries = Dict[str, List[str]]


def weighted_average(
    span_metrics: Dict[str, Dict[str, float]],
    frequencies: Dict[str, Counter],
) -> Dict[str, float]:
    weighted_average = {}
    for spans_key, span_type_dict in span_metrics.items():
        weighted_average[spans_key] = np.average(
            list(span_type_dict.values()),
            weights=list(frequencies[spans_key].values()),
        )

    return weighted_average


class SpanAnalyzer:
    def __init__(self, docs: List[Doc]):
        self.docs = docs
        self.p_corpus = self._get_distribution(self.docs, normalize=True)
        self.keys = self._get_all_keys()

    @property
    def frequency(self) -> Dict[str, Counter]:
        """Number of spans for a span type in the dataset's training corpus.

        Frequency tends to be positively correlated with performance. However,
        architectural choices (like the use of transfer learning) can reduce
        data requirements for ML models. For such architectures, there is a
        small correlation between frequency and performance.
        """
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
    def length(self) -> Dict[str, Dict[str, float]]:
        """Geometric mean of the spans' lengths in tokens.

        Traditional CRF models tend to perform poorly at the identification of
        long spans due to their strict Markov assumption. Architectures that
        rely on such assumptions should follow the same pattern. On the other
        hand, LSTMs or Transformers should do better on long spans.
        """
        _length = {}
        for doc in self.docs:
            for spans_key in list(doc.spans.keys()):
                if spans_key not in _length:
                    _length[spans_key] = {}
                for span in doc.spans[spans_key]:
                    if span.label_ is None:
                        continue
                    elif span.label_ not in _length[spans_key]:
                        _length[spans_key][span.label_] = []
                    else:
                        _length[spans_key][span.label_].append(len(span))

        # Compute for the geometric mean for each span type
        length = {}
        for spans_key, span_type_dict in _length.items():
            length[spans_key] = {}
            for span_type, lengths in span_type_dict.items():
                length[spans_key][span_type] = gmean(lengths)

        return length

    @property
    def span_distinctiveness(self) -> Dict[str, float]:
        """Distinctiveness of the span compared to the corpus.

        Measures how distinct the text comprising the spans compared to the
        rest of the corpus. It is defined as the KL divergence D(P_span || P),
        where P is the unigram word distribution of the corpus, and P_span as
        the unigram distribution of tokens within the span.

        High values indicate that different words are used inside spans
        compared to the rest of the text, whereas low values indicate that the
        word distribution is similar inside and outside of spans.

        This property is positively correlated with model performance. Spans
        with high distinctiveness should be able to rely more heavily on local
        features, as each token carries information about span membership. Low
        span distrinctivess then calls for sequence information.
        """
        p_spans: Dict[str, Dict[str, FrequencyDist]] = {}
        for key in self.keys:
            p_spans[key] = {}
            spans_per_type = self._get_all_spans_in_key(key)
            for span_type, spans in spans_per_type.items():
                p_span = self._get_distribution(spans, normalize=True)
                p_spans[key][span_type] = p_span

        # Compute value for each spans_key
        span_distincts: Dict[str, Dict[str, float]] = {}
        for spans_key, freq_per_type in p_spans.items():
            span_distincts[spans_key] = {}
            for span_type, frequency_dist in freq_per_type.items():
                span_distincts[spans_key][span_type] = self._get_kl_divergence(
                    frequency_dist, self.p_corpus
                )
        return span_distincts

    @property
    def boundary_distinctiveness(self):
        """Distinctiveness of the boundaries compared to the corpus.

        Measures how distinctive the starts and ends of spans are. It is
        formalized as the KL-divergence D(P_bounds || P) where P is the unigram
        word distribution of the corpus, and P_bounds as the unigram
        distribution of the boundary tokens.

        This property is positively correlated with model performance. High
        values mean that the start and end points of spans are easy to spot,
        while low values indicate smooth transitions.
        """
        p_bounds: Dict[str, Dict[str, FrequencyDist]] = {}
        for key in self.keys:
            p_bounds[key] = {}
            span_boundaries_per_type = self._get_all_boundaries_in_key(key)
            for span_type, span_bounds in span_boundaries_per_type.items():
                bounds = span_bounds["start"] + span_bounds["end"]
                p_bound = self._get_distribution(bounds, normalize=True, unigrams=True)
                p_bounds[key][span_type] = p_bound

        # Compute value for each spans_key
        bound_distincts: Dict[str, Dict[str, float]] = {}
        for spans_key, freq_per_type in p_bounds.items():
            bound_distincts[spans_key] = {}
            for span_type, frequency_dist in freq_per_type.items():
                bound_distincts[spans_key][span_type] = self._get_kl_divergence(
                    frequency_dist, self.p_corpus
                )

        return bound_distincts

    def _get_all_keys(self) -> Set[str]:
        """Get all spans_key in the corpus"""
        return set([key for doc in self.docs for key in list(doc.spans.keys())])

    def _get_all_spans_in_key(self, spans_key: str) -> Dict[str, List[Span]]:
        """Get all spans given a specified spans_key

        Returns a dictionary where the keys are the span types and
        the values are the list of Spans within that span type.
        """
        spans: Dict[str, List[Span]] = {}
        for doc in self.docs:
            for span in doc.spans[spans_key]:
                if span.label_ not in spans:
                    spans[span.label_] = []
                else:
                    spans[span.label_].append(span)

        return spans

    def _get_all_boundaries_in_key(self, spans_key: str) -> Dict[str, SpanBoundaries]:
        """Get the boundary tokens for all spans in a spans_key

        Returns a dictionary where the keys are the span types and
        the values are a tuple of lists for the start and end span
        boundaries.
        """
        bounds: Dict[str, SpanBoundaries] = {}
        for doc in self.docs:
            for span in doc.spans[spans_key]:
                if span.label_ not in bounds:
                    bounds[span.label_] = {"start": [], "end": []}
                else:
                    # Get span boundaries
                    span_bound_start_idx = span.start - 1
                    if span_bound_start_idx >= 0:
                        bounds[span.label_]["start"].append(doc[span_bound_start_idx])

                    span_bound_end_idx = span.end + 1
                    if span_bound_end_idx < len(doc):
                        bounds[span.label_]["end"].append(doc[span_bound_end_idx])
        return bounds

    def _get_distribution(
        self,
        texts: Union[List[Doc], List[SpanGroup], List[Token]],
        normalize: bool = True,
        unigrams: bool = False,
    ) -> Counter:
        """Get each word's sample frequency given a list of text"""
        word_counts = Counter()
        for text in texts:
            if unigrams:
                word_counts[self._normalize_text(text.text)] += 1
            else:
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

    def _normalize_text(self, text: str) -> str:
        # The source code from the paper converts all digits into a single
        # value, '0'. Perhaps this is a way to "normalize" all numerical value
        # I'm not sure if this is generalizable so I'll comment it out for the meantime.

        # pattern = re.compile("[0-9]")
        # text = re.sub(pattern, "0", text)
        text = text.lower().replace("``", '"').replace("''", '"')
        return text
