"""LaMP-5 ROUGE using the same backend and defaults as official LaMP evaluation.

Official LaMP calls ``evaluate.load('rouge')`` and reports ``rouge1`` and
``rougeL``. Hugging Face's metric delegates to ``rouge_score`` with
``use_stemmer=False``. Candidate filtering needs per-example scores, while
final reporting uses the same BootstrapAggregator as the official wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence

from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu as sacrebleu_corpus_bleu


_ROUGE_TYPES = ("rouge1", "rougeL")
_SCORER = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=False)


def score(prediction: str, reference: str) -> dict[str, float]:
    result = _SCORER.score(reference.strip(), prediction.strip())
    return {
        "rouge_1": result["rouge1"].fmeasure,
        "rouge_l": result["rougeL"].fmeasure,
    }


def corpus_score(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    return {
        metric: interval["mid"]
        for metric, interval in corpus_score_with_ci(predictions, references).items()
    }


def corpus_bleu(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float | int | list[float] | list[int]]:
    """Return SacreBLEU with the same defaults used by HYDRA's evaluator.

    SacreBLEU is a corpus-level metric on a 0--100 scale. The returned
    statistics make the exact calculation auditable without changing the
    official LaMP ROUGE metrics used for Ranker supervision and selection.
    """
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and have equal length")
    result = sacrebleu_corpus_bleu(
        [prediction.strip() for prediction in predictions],
        [[reference.strip() for reference in references]],
    )
    return {
        "score": float(result.score),
        "counts": [int(value) for value in result.counts],
        "totals": [int(value) for value in result.totals],
        "precisions": [float(value) for value in result.precisions],
        "bp": float(result.bp),
        "sys_len": int(result.sys_len),
        "ref_len": int(result.ref_len),
    }


def corpus_score_with_ci(
    predictions: Sequence[str], references: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Return official-compatible ROUGE together with bootstrap intervals.

    LaMP reports the bootstrap midpoint.  Keeping the lower/upper bounds in the
    experiment artifact makes differences on small development sets easier to
    interpret without changing the primary metric implementation.
    """
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and have equal length")
    aggregator = scoring.BootstrapAggregator()
    for prediction, reference in zip(predictions, references):
        aggregator.add_scores(_SCORER.score(reference.strip(), prediction.strip()))
    result = aggregator.aggregate()
    return {
        "rouge_1": {
            "low": float(result["rouge1"].low.fmeasure),
            "mid": float(result["rouge1"].mid.fmeasure),
            "high": float(result["rouge1"].high.fmeasure),
        },
        "rouge_l": {
            "low": float(result["rougeL"].low.fmeasure),
            "mid": float(result["rougeL"].mid.fmeasure),
            "high": float(result["rougeL"].high.fmeasure),
        },
    }
