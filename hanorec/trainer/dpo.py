"""LLaMA-Factory DPO trainer implementing HaRS and NoDO."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from llamafactory.train.dpo.trainer import CustomDPOTrainer

from ..nodo.hooks import perturb_lora_weights


_METADATA_KEYS = {
    "hanorec_hardness",
    "hanorec_positive_item_ids",
    "hanorec_negative_item_ids",
}


def _responsiveness(reward_gaps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Torch equivalent of paper equation (8), treated as a batch statistic."""

    values = reward_gaps.reshape(-1)
    mean_gap = values.mean()
    fallback = values.abs().mean().clamp_min(eps)
    scale = torch.where(mean_gap.abs() > eps, mean_gap, fallback)
    normalized = values / scale
    if normalized.numel() > 2:
        trimmed = normalized.sort().values[1:-1]
    else:
        trimmed = normalized
    return torch.sigmoid(trimmed.mean()) / torch.sigmoid(normalized.mean())


class HaNoRecDPOTrainer(CustomDPOTrainer):
    """DPO with offline HaRS hardness and transient NoDO LoRA perturbations."""

    def _split_hanorec_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if "hanorec_hardness" not in batch:
            raise ValueError(
                "HaNoRec requires hanorec_hardness; run scripts/prepare_hanorec.py first."
            )
        clean_batch = {key: value for key, value in batch.items() if key not in _METADATA_KEYS}
        return clean_batch, batch["hanorec_hardness"]

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean_batch, hardness = self._split_hanorec_batch(batch)

        if train_eval == "train":
            with perturb_lora_weights(
                model,
                sigma=self.finetuning_args.hanorec_noise_sigma,
            ):
                policy_outputs = super().concatenated_forward(model, clean_batch)
        else:
            policy_outputs = super().concatenated_forward(model, clean_batch)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = policy_outputs
        reference_chosen_logps, reference_rejected_logps = super().compute_reference_log_probs(
            model, clean_batch
        )
        if reference_chosen_logps is None or reference_rejected_logps is None:
            raise RuntimeError("HaNoRec requires an unperturbed reference model")

        hardness = hardness.to(
            device=policy_chosen_logps.device,
            dtype=policy_chosen_logps.dtype,
        ).reshape(-1)
        if hardness.numel() != policy_chosen_logps.numel():
            raise ValueError("HaNoRec hardness must contain one value per preference pair")
        if not torch.isfinite(hardness).all() or (hardness <= 0).any():
            raise ValueError("HaNoRec hardness values must be finite and positive")

        preference_logits = (
            policy_chosen_logps
            - reference_chosen_logps
            - policy_rejected_logps
            + reference_rejected_logps
        )
        with torch.no_grad():
            local_reward_gaps = self.beta * preference_logits.detach()
            reward_gaps = self.accelerator.gather(local_reward_gaps)
            if reward_gaps.numel() < 3:
                raise ValueError("HaNoRec Eq. (8) requires a global mini-batch of at least 3 samples")
            responsiveness = _responsiveness(reward_gaps)
            betas = (
                self.beta * responsiveness * hardness.detach()
            ).clamp_min(self.finetuning_args.hanorec_beta_floor)

        scaled_logits = betas * preference_logits
        losses = -(1.0 - self.label_smoothing) * F.logsigmoid(scaled_logits)
        if self.label_smoothing > 0:
            losses -= self.label_smoothing * F.logsigmoid(-scaled_logits)

        chosen_rewards = betas * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
        rejected_rewards = betas * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses = losses + self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.mean().item(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().item(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            f"{prefix}logps/chosen": policy_chosen_logps.mean().item(),
            f"{prefix}logps/rejected": policy_rejected_logps.mean().item(),
            f"{prefix}logits/chosen": policy_chosen_logits.mean().item(),
            f"{prefix}logits/rejected": policy_rejected_logits.mean().item(),
            f"{prefix}hanorec/responsiveness": responsiveness.item(),
            f"{prefix}hanorec/beta": betas.mean().item(),
            f"{prefix}hanorec/hardness": hardness.mean().item(),
        }
        return losses.mean(), metrics
