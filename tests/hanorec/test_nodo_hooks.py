from __future__ import annotations

import unittest

from hanorec.nodo.hooks import iter_active_lora_pairs, perturb_lora_weights


class _FakeLoraLayer:
    def __init__(self) -> None:
        self.lora_A = {"default": object(), "unused": object()}
        self.lora_B = {"default": object(), "unused": object()}
        self.active_adapters = ["default"]


class _FakeModel:
    def named_modules(self):
        yield "", self
        yield "decoder.layer", _FakeLoraLayer()


class _FakeEmptyModel:
    def named_modules(self):
        yield "", self


class LoraDiscoveryTest(unittest.TestCase):
    def test_only_active_adapter_pairs_are_discovered(self) -> None:
        pairs = list(iter_active_lora_pairs(_FakeModel()))

        self.assertEqual(len(pairs), 1)
        name, adapter, lora_a, lora_b = pairs[0]
        self.assertEqual(name, "decoder.layer")
        self.assertEqual(adapter, "default")
        self.assertIsNotNone(lora_a)
        self.assertIsNotNone(lora_b)

    def test_missing_lora_layers_fail_before_training(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No active LoRA"):
            with perturb_lora_weights(_FakeEmptyModel(), sigma=0.1):
                pass


try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - lightweight CPU environment
    torch = None
    F = None


@unittest.skipIf(torch is None, "PyTorch is required for LoRA hook integration tests")
class LoraPerturbationTest(unittest.TestCase):
    class FakePeftLayer(torch.nn.Module if torch is not None else object):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 2, bias=False)}
            )
            self.active_adapters = ["default"]
            with torch.no_grad():
                self.lora_A["default"].weight.copy_(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0]])
                )
                self.lora_B["default"].weight.copy_(
                    torch.tensor([[0.5, 0.0], [0.0, 0.5]])
                )

        def forward(self, values):
            return values + self.lora_B["default"](self.lora_A["default"](values))

    def test_hook_matches_explicit_perturbed_matrix_product(self) -> None:
        model = self.FakePeftLayer()
        values = torch.tensor([[1.0, -2.0]])
        clean = model(values)
        generator = torch.Generator().manual_seed(13)

        with perturb_lora_weights(
            model, sigma=0.05, generator=generator
        ) as records:
            actual = model(values)
            record = records[0]
            a_prime = F.linear(
                values,
                model.lora_A["default"].weight + record.noise_a,
            )
            expected = values + F.linear(
                a_prime,
                model.lora_B["default"].weight + record.noise_b,
            )

        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(model(values), clean))

    def test_gradients_flow_and_hooks_are_removed(self) -> None:
        model = self.FakePeftLayer()
        values = torch.tensor([[0.2, 0.7]], requires_grad=True)
        before_a = len(model.lora_A["default"]._forward_hooks)
        before_b = len(model.lora_B["default"]._forward_hooks)

        with perturb_lora_weights(model, sigma=0.1):
            model(values).sum().backward()

        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(model.lora_A["default"].weight.grad)
        self.assertIsNotNone(model.lora_B["default"].weight.grad)
        self.assertEqual(len(model.lora_A["default"]._forward_hooks), before_a)
        self.assertEqual(len(model.lora_B["default"]._forward_hooks), before_b)


if __name__ == "__main__":
    unittest.main()
