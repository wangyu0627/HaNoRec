"""Dependency-light implementations of HaRS equations (5), (6), (8), and (11)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _finite_values(values: Sequence[float], name: str) -> list[float]:
    checked = [float(value) for value in values]
    if not checked:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(value) for value in checked):
        raise ValueError(f"{name} must contain only finite values")
    return checked


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softmax(values: Sequence[float]) -> list[float]:
    """Return a numerically stable probability distribution."""

    checked = _finite_values(values, "values")
    offset = max(checked)
    weights = [math.exp(value - offset) for value in checked]
    total = sum(weights)
    return [weight / total for weight in weights]


def probability_distance(chosen: Sequence[float], rejected: Sequence[float]) -> float:
    """Compute the L2 distance between aligned chosen/rejected distributions."""

    chosen_values = _finite_values(chosen, "chosen")
    rejected_values = _finite_values(rejected, "rejected")
    if len(chosen_values) != len(rejected_values):
        raise ValueError("chosen and rejected must have equal length")
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(chosen_values, rejected_values, strict=True))
    )


def normalize_hardness(deltas: Sequence[float]) -> list[float]:
    """Apply equation (6): sigmoid(delta) divided by sigmoid(mean delta)."""

    checked = _finite_values(deltas, "deltas")
    if any(delta < 0 for delta in checked):
        raise ValueError("deltas must be non-negative")
    denominator = _sigmoid(sum(checked) / len(checked))
    return [_sigmoid(delta) / denominator for delta in checked]


def model_responsiveness(reward_gaps: Sequence[float], eps: float = 1e-8) -> float:
    """Estimate batch responsiveness with mean normalization and extreme filtering."""

    checked = _finite_values(reward_gaps, "reward_gaps")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    mean_gap = sum(checked) / len(checked)
    if abs(mean_gap) > eps:
        scale = mean_gap
    else:
        scale = max(sum(abs(value) for value in checked) / len(checked), eps)

    normalized = [value / scale for value in checked]
    trimmed = sorted(normalized)[1:-1] if len(normalized) > 2 else normalized
    trimmed_mean = sum(trimmed) / len(trimmed)
    normalized_mean = sum(normalized) / len(normalized)
    return _sigmoid(trimmed_mean) / _sigmoid(normalized_mean)


def dynamic_beta(
    hardness: Sequence[float],
    responsiveness: float,
    beta0: float,
    floor: float = 1e-6,
) -> list[float]:
    """Compute per-example equation (11) beta values with a numerical floor."""

    checked = _finite_values(hardness, "hardness")
    if any(value < 0 for value in checked):
        raise ValueError("hardness must be non-negative")
    if not math.isfinite(responsiveness) or responsiveness < 0:
        raise ValueError("responsiveness must be finite and non-negative")
    if not math.isfinite(beta0) or beta0 <= 0:
        raise ValueError("beta0 must be finite and positive")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    return [max(floor, beta0 * responsiveness * value) for value in checked]
