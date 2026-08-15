from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


class RuntimeAssetsTest(unittest.TestCase):
    def test_all_current_dataset_tasks_have_paper_aligned_configs(self) -> None:
        for dataset in ("microlens", "netflix", "movielens"):
            for hit in (1, 3):
                with self.subTest(dataset=dataset, hit=hit):
                    path = ROOT / "configs" / "hanorec" / f"{dataset}_hit{hit}.yaml"
                    self.assertTrue(path.is_file())
                    config = _simple_yaml(path)
                    self.assertEqual(config["stage"], "dpo")
                    self.assertEqual(config["use_hanorec"], "true")
                    self.assertEqual(config["pref_beta"], "0.1")
                    self.assertEqual(config["lora_rank"], "8")
                    self.assertEqual(config["learning_rate"], "1.0e-4")
                    self.assertEqual(config["num_train_epochs"], "5.0")
                    self.assertEqual(config["gradient_accumulation_steps"], "8")
                    self.assertGreaterEqual(int(config["per_device_train_batch_size"]), 3)
                    self.assertEqual(config["dataloader_drop_last"], "true")
                    expected_sigma = "0.05" if dataset == "microlens" else "0.1"
                    self.assertEqual(config["hanorec_noise_sigma"], expected_sigma)

    def test_dataset_info_registers_six_derived_pairwise_files(self) -> None:
        info_path = ROOT / "LLaMA-Factory" / "data" / "dataset_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for dataset in ("microlens", "netflix", "movielens"):
            for hit in (1, 3):
                name = f"{dataset}_vl_train_hanorec_{hit}"
                entry = info[name]
                self.assertTrue(entry["ranking"])
                self.assertEqual(entry["columns"]["hardness"], "hardness")
                self.assertIn("artifacts/hanorec", entry["file_name"])

    def test_cross_platform_prepare_and_train_entrypoints_exist(self) -> None:
        prepare = ROOT / "scripts" / "prepare_hanorec.py"
        train = ROOT / "scripts" / "train_hanorec.py"
        main_source = (ROOT / "main_mllm.py").read_text(encoding="utf-8")

        self.assertTrue(prepare.is_file())
        self.assertTrue(train.is_file())
        self.assertIn("hanorec.cli.build_preferences", prepare.read_text(encoding="utf-8"))
        self.assertIn("hanorec.cli.build_hardness", prepare.read_text(encoding="utf-8"))
        self.assertIn("llamafactory.cli", train.read_text(encoding="utf-8"))
        self.assertIn("scripts/train_hanorec.py", main_source)
        self.assertIn('args.training_mode == "sft"', main_source)


if __name__ == "__main__":
    unittest.main()
