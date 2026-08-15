"""Temporary LoRA perturbation hooks implementing the NoDO forward pass."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerturbationRecord:
    """The sampled noise for one active LoRA adapter pair."""

    module_name: str
    adapter_name: str
    noise_a: Any
    noise_b: Any


def _active_adapter_names(module: Any) -> list[str]:
    adapters = getattr(module, "active_adapters", None)
    if adapters is None:
        adapters = getattr(module, "active_adapter", None)
    if isinstance(adapters, str):
        return [adapters]
    if adapters is None:
        return []
    return list(adapters)


def iter_active_lora_pairs(model: Any) -> Iterator[tuple[str, str, Any, Any]]:
    """Yield active PEFT LoRA A/B modules without importing PEFT."""

    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        for adapter_name in _active_adapter_names(module):
            if adapter_name in lora_a and adapter_name in lora_b:
                yield (
                    module_name,
                    adapter_name,
                    lora_a[adapter_name],
                    lora_b[adapter_name],
                )


def _sample_like(weight: Any, sigma: float, generator: Any | None) -> Any:
    import torch

    return torch.randn(
        weight.shape,
        device=weight.device,
        dtype=weight.dtype,
        generator=generator,
    ) * float(sigma)


def _noise_hook(noise: Any):
    import torch.nn.functional as functional

    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if not inputs:
            raise RuntimeError("A LoRA linear module received no positional input")
        return output + functional.linear(inputs[0], noise, None)

    return hook


@contextmanager
def perturb_lora_weights(
    model: Any,
    *,
    sigma: float,
    generator: Any | None = None,
) -> Iterator[list[PerturbationRecord]]:
    """Apply A'=A+eps_A and B'=B+eps_B for one differentiable forward scope.

    The actual parameters are never mutated. Forward hooks add the equivalent
    linear noise terms and are removed even if the model forward raises.
    """

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    pairs = list(iter_active_lora_pairs(model))
    if not pairs:
        raise RuntimeError(
            "No active LoRA A/B module pairs were found; NoDO requires PEFT LoRA."
        )

    handles: list[Any] = []
    records: list[PerturbationRecord] = []
    try:
        for module_name, adapter_name, lora_a, lora_b in pairs:
            noise_a = _sample_like(lora_a.weight, sigma, generator)
            noise_b = _sample_like(lora_b.weight, sigma, generator)
            handles.append(lora_a.register_forward_hook(_noise_hook(noise_a)))
            handles.append(lora_b.register_forward_hook(_noise_hook(noise_b)))
            records.append(
                PerturbationRecord(
                    module_name=module_name,
                    adapter_name=adapter_name,
                    noise_a=noise_a,
                    noise_b=noise_b,
                )
            )
        yield records
    finally:
        for handle in reversed(handles):
            handle.remove()
