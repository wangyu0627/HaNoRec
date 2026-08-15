from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "data"


class LlamaFactoryMetadataIntegrationTest(unittest.TestCase):
    def test_parser_declares_hanorec_columns(self) -> None:
        source = (DATA_ROOT / "parser.py").read_text(encoding="utf-8")
        self.assertIn("hardness: Optional[str]", source)
        self.assertIn("positive_item_ids: Optional[str]", source)
        self.assertIn("negative_item_ids: Optional[str]", source)
        self.assertIn('"hardness", "positive_item_ids", "negative_item_ids"', source)

    def test_converter_preserves_hanorec_fields(self) -> None:
        source = (DATA_ROOT / "converter.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"_hanorec_hardness"'), 2)
        self.assertGreaterEqual(source.count('"_hanorec_positive_item_ids"'), 2)
        self.assertGreaterEqual(source.count('"_hanorec_negative_item_ids"'), 2)

    def test_pairwise_processor_and_collator_forward_metadata(self) -> None:
        processor = (DATA_ROOT / "processor" / "pairwise.py").read_text(
            encoding="utf-8"
        )
        collator = (DATA_ROOT / "collator.py").read_text(encoding="utf-8")
        self.assertIn('model_inputs["hanorec_hardness"]', processor)
        self.assertIn("extract_hanorec_metadata(features)", collator)
        self.assertIn('batch["hanorec_hardness"]', collator)
        self.assertNotIn(
            "from hanorec.data.metadata import", collator.split("class PairwiseDataCollator", 1)[0]
        )
        self.assertIn('if any("hanorec_hardness" in feature for feature in features):', collator)
    def test_workflow_selects_hanorec_trainer_only_when_enabled(self) -> None:
        workflow = (
            ROOT
            / "LLaMA-Factory"
            / "src"
            / "llamafactory"
            / "train"
            / "dpo"
            / "workflow.py"
        ).read_text(encoding="utf-8")
        arguments = (
            ROOT
            / "LLaMA-Factory"
            / "src"
            / "llamafactory"
            / "hparams"
            / "finetuning_args.py"
        ).read_text(encoding="utf-8")
        trainer_path = ROOT / "hanorec" / "trainer" / "dpo.py"

        self.assertIn("finetuning_args.use_hanorec", workflow)
        self.assertIn("HaNoRecDPOTrainer", workflow)
        self.assertIn("use_hanorec: bool", arguments)
        self.assertIn("hanorec_noise_sigma: float", arguments)
        self.assertTrue(trainer_path.is_file())
        trainer_source = trainer_path.read_text(encoding="utf-8")
        self.assertIn("class HaNoRecDPOTrainer", trainer_source)
        self.assertIn("self.accelerator.gather", trainer_source)



if __name__ == "__main__":
    unittest.main()
