"""Target-blind structural summaries of a user's historical titles."""

from __future__ import annotations

import re
import statistics


WORDS = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def style_summary(titles: list[str]) -> str:
    """Return a compact, stable style direction derived only from history."""
    clean = [str(title).strip() for title in titles if str(title).strip()]
    if not clean:
        return "No reliable historical-title style statistics are available."
    lengths = [len(WORDS.findall(title)) for title in clean]
    median_length = statistics.median(lengths)
    colon_rate = sum(":" in title for title in clean) / len(clean)
    question_rate = sum("?" in title for title in clean) / len(clean)
    acronym_rate = sum(
        any(len(token) >= 2 and token.isupper() for token in WORDS.findall(title))
        for title in clean
    ) / len(clean)
    return (
        f"Historical titles typically contain {median_length:g} words; "
        f"{colon_rate:.0%} use a colon, {question_rate:.0%} are questions, and "
        f"{acronym_rate:.0%} contain an uppercase acronym. Preserve these structural "
        "tendencies only when they fit the current abstract."
    )

