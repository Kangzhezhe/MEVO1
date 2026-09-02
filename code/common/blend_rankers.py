"""Target-blind score calibration and blending for two candidate rankers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common.utils import read_jsonl, write_jsonl


def _standardized_scores(row: dict) -> dict[str, float]:
    values = {
        str(candidate["candidate_id"]): float(candidate["ranker_score"])
        for candidate in row["ranked_candidates"]
    }
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    scale = math.sqrt(variance) or 1.0
    return {key: (value - mean) / scale for key, value in values.items()}


def blend(primary_path: Path, secondary_path: Path, destination: Path, alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    primary = {str(row["sample_id"]): row for row in read_jsonl(primary_path)}
    secondary = {str(row["sample_id"]): row for row in read_jsonl(secondary_path)}
    if set(primary) != set(secondary):
        raise ValueError("Ranker prediction files must contain identical sample IDs")
    output = []
    for sample_id in sorted(primary):
        first = primary[sample_id]
        second = secondary[sample_id]
        first_scores = _standardized_scores(first)
        second_scores = _standardized_scores(second)
        if set(first_scores) != set(second_scores):
            raise ValueError(f"Candidate IDs differ for sample={sample_id}")
        candidates = {str(row["candidate_id"]): row for row in first["ranked_candidates"]}
        ranked = []
        for candidate_id, candidate in candidates.items():
            blended_score = (
                (1.0 - alpha) * first_scores[candidate_id]
                + alpha * second_scores[candidate_id]
            )
            ranked.append({**candidate, "ranker_score": blended_score})
        ranked.sort(key=lambda row: (-float(row["ranker_score"]), str(row["candidate_id"])))
        output.append({
            "sample_id": sample_id,
            "selected_id": str(ranked[0]["candidate_id"]),
            "prediction": str(ranked[0]["text"]),
            "ranked_candidates": ranked,
            "blend": {
                "primary": str(primary_path),
                "secondary": str(secondary_path),
                "secondary_weight": alpha,
                "calibration": "within_group_zscore",
            },
        })
    write_jsonl(destination, output)
    print(f"blended predictions for {len(output)} samples -> {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend two target-blind ranker score files")
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args()
    blend(args.primary, args.secondary, args.output, args.alpha)


if __name__ == "__main__":
    main()
