"""Pure-Python reference implementation of the HaNoRec DPO equations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..hars.math import dynamic_beta, model_responsiveness


def _values(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def hanorec_dpo_terms(
    *,
    policy_chosen: Sequence[float],
    policy_rejected: Sequence[float],
    reference_chosen: Sequence[float],
    reference_rejected: Sequence[float],
    hardness: Sequence[float],
    beta0: float,
    beta_floor: float = 1e-6,
) -> dict[str, list[float] | float]:
    """Compute equations (7), (8), (11), and (12) for one batch."""

    pi_chosen = _values(policy_chosen, "policy_chosen")
    pi_rejected = _values(policy_rejected, "policy_rejected")
    ref_chosen = _values(reference_chosen, "reference_chosen")
    ref_rejected = _values(reference_rejected, "reference_rejected")
    hardness_values = _values(hardness, "hardness")
    lengths = {
        len(pi_chosen),
        len(pi_rejected),
        len(ref_chosen),
        len(ref_rejected),
        len(hardness_values),
    }
    if len(lengths) != 1:
        raise ValueError("All HaNoRec batch inputs must have equal length")

    preference_logits = [
        (chosen - reference_chosen_value) - (rejected - reference_rejected_value)
        for chosen, rejected, reference_chosen_value, reference_rejected_value in zip(
            pi_chosen,
            pi_rejected,
            ref_chosen,
            ref_rejected,
            strict=True,
        )
    ]
    base_reward_gaps = [float(beta0) * value for value in preference_logits]
    responsiveness = model_responsiveness(base_reward_gaps)
    betas = dynamic_beta(hardness_values, responsiveness, beta0, beta_floor)
    chosen_rewards = [
        beta * (chosen - reference)
        for beta, chosen, reference in zip(betas, pi_chosen, ref_chosen, strict=True)
    ]
    rejected_rewards = [
        beta * (rejected - reference)
        for beta, rejected, reference in zip(betas, pi_rejected, ref_rejected, strict=True)
    ]
    margins = [
        chosen - rejected
        for chosen, rejected in zip(chosen_rewards, rejected_rewards, strict=True)
    ]
    losses = [
        _softplus(-beta * logit)
        for beta, logit in zip(betas, preference_logits, strict=True)
    ]
    return {
        "responsiveness": responsiveness,
        "betas": betas,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_margins": margins,
        "losses": losses,
    }
