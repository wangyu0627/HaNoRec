"""Dependency-light validation helpers for HaNoRec batch metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


_FEATURE_KEYS = (
    "hanorec_hardness",
    "hanorec_positive_item_ids",
    "hanorec_negative_item_ids",
)


def extract_hanorec_metadata(
    features: Sequence[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Validate and extract optional metadata from pairwise features."""

    if not features:
        return {}
    present = [all(key in feature for key in _FEATURE_KEYS) for feature in features]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError("HaNoRec metadata must be present on every sample in a batch")

    hardness: list[float] = []
    positives: list[list[int]] = []
    negatives: list[list[int]] = []
    for index, feature in enumerate(features):
        value = float(feature["hanorec_hardness"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Sample {index} hardness must be finite and positive")
        positive = [int(item_id) for item_id in feature["hanorec_positive_item_ids"]]
        negative = [int(item_id) for item_id in feature["hanorec_negative_item_ids"]]
        if not positive or len(positive) != len(negative):
            raise ValueError(f"Sample {index} item-id pairs must be aligned and non-empty")
        hardness.append(value)
        positives.append(positive)
        negatives.append(negative)

    return {
        "hardness": hardness,
        "positive_item_ids": positives,
        "negative_item_ids": negatives,
    }


def pad_item_ids(rows: Sequence[Sequence[int]], pad_value: int = -1) -> list[list[int]]:
    """Pad variable-length item-id lists without introducing a valid item id."""

    if not rows:
        return []
    width = max(len(row) for row in rows)
    return [
        [int(value) for value in row] + [int(pad_value)] * (width - len(row))
        for row in rows
    ]
