"""Offline fused-embedding retrieval and HaRS hardness computation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .math import normalize_hardness, probability_distance, softmax


def _as_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def fuse_embeddings(text_embeddings: Any, visual_embeddings: Any) -> np.ndarray:
    """Normalize modalities, add them, and normalize the fused item vectors."""

    text = _as_matrix(text_embeddings, "text_embeddings")
    visual = _as_matrix(visual_embeddings, "visual_embeddings")
    if text.shape != visual.shape:
        raise ValueError("text and visual embeddings must have identical shape")
    fused = _normalize_rows(text) + _normalize_rows(visual)
    return _normalize_rows(fused)


def topk_neighbors(
    item_ids: Any,
    embeddings: Any,
    *,
    k: int,
    block_size: int = 256,
) -> dict[int, dict[str, list[Any]]]:
    """Retrieve cosine Top-K neighbors in bounded-memory anchor blocks."""

    ids = np.asarray(item_ids, dtype=np.int64)
    matrix = _normalize_rows(_as_matrix(embeddings, "embeddings"))
    if ids.ndim != 1 or len(ids) != matrix.shape[0]:
        raise ValueError("item_ids must be rank-1 and align with embeddings")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("item_ids must be unique")
    if not 0 < k < len(ids):
        raise ValueError("k must be positive and smaller than the catalog")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    result: dict[int, dict[str, list[Any]]] = {}
    for start in range(0, len(ids), block_size):
        stop = min(start + block_size, len(ids))
        scores = matrix[start:stop] @ matrix.T
        for local_index, anchor_index in enumerate(range(start, stop)):
            row = scores[local_index]
            row[anchor_index] = -np.inf
            selected = np.argpartition(-row, k - 1)[:k]
            selected = sorted(selected.tolist(), key=lambda index: (-float(row[index]), int(ids[index])))
            anchor_id = int(ids[anchor_index])
            result[anchor_id] = {
                "item_ids": [int(ids[index]) for index in selected],
                "scores": [float(row[index]) for index in selected],
            }
    return result


def _pair_delta(
    positive_item_id: int,
    negative_item_id: int,
    neighbors: dict[int, dict[str, list[Any]]],
) -> float:
    try:
        positive_scores = neighbors[int(positive_item_id)]["scores"]
        negative_scores = neighbors[int(negative_item_id)]["scores"]
    except KeyError as error:
        raise ValueError(f"Missing neighbor scores for item {error.args[0]}") from error
    return probability_distance(softmax(positive_scores), softmax(negative_scores))


def compute_sample_deltas(
    examples: Sequence[dict[str, Any]],
    neighbors: dict[int, dict[str, list[Any]]],
) -> list[float]:
    deltas: list[float] = []
    for index, example in enumerate(examples):
        positives = [int(value) for value in example.get("positive_item_ids", [])]
        negatives = [int(value) for value in example.get("negative_item_ids", [])]
        if not positives or len(positives) != len(negatives):
            raise ValueError(f"Sample {index} must contain aligned non-empty item-id pairs")
        pair_deltas = [
            _pair_delta(positive, negative, neighbors)
            for positive, negative in zip(positives, negatives, strict=True)
        ]
        deltas.append(sum(pair_deltas) / len(pair_deltas))
    if not deltas:
        raise ValueError("examples must be non-empty")
    return deltas


def compute_sample_hardness(
    examples: Sequence[dict[str, Any]],
    neighbors: dict[int, dict[str, list[Any]]],
) -> list[float]:
    return normalize_hardness(compute_sample_deltas(examples, neighbors))


def attach_hardness(
    examples: Sequence[dict[str, Any]],
    hardness: Sequence[float],
) -> list[dict[str, Any]]:
    values = [float(value) for value in hardness]
    if len(examples) != len(values):
        raise ValueError("examples and hardness must have equal length")
    if not np.isfinite(values).all() or any(value < 0 for value in values):
        raise ValueError("hardness values must be finite and non-negative")
    attached: list[dict[str, Any]] = []
    for example, value in zip(examples, values, strict=True):
        copied = dict(example)
        copied["hardness"] = value
        attached.append(copied)
    return attached
