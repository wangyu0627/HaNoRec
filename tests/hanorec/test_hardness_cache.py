import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hanorec.hars.cache import (
    StaleArtifactError,
    build_embedding_manifest,
    build_manifest,
    validate_manifest,
)
from hanorec.hars.hardness import (
    attach_hardness,
    compute_sample_hardness,
    fuse_embeddings,
    topk_neighbors,
)


class EmbeddingMathTest(unittest.TestCase):
    def test_missing_visual_vector_preserves_normalized_text(self):
        text = np.asarray([[3.0, 4.0]])
        visual = np.zeros_like(text)

        fused = fuse_embeddings(text, visual)

        np.testing.assert_allclose(fused, [[0.6, 0.8]])

    def test_modalities_are_normalized_before_fusion(self):
        text = np.asarray([[10.0, 0.0]])
        visual = np.asarray([[0.0, 2.0]])

        fused = fuse_embeddings(text, visual)

        expected = np.asarray([[1.0, 1.0]]) / np.sqrt(2.0)
        np.testing.assert_allclose(fused, expected)

    def test_topk_excludes_anchor(self):
        item_ids = np.asarray([10, 20, 30])
        embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

        neighbors = topk_neighbors(item_ids, embeddings, k=2)

        self.assertNotIn(10, neighbors[10]["item_ids"])
        self.assertEqual(neighbors[10]["item_ids"][0], 20)
        self.assertEqual(len(neighbors[10]["scores"]), 2)


class HardnessTest(unittest.TestCase):
    def test_easy_pair_receives_higher_hardness_than_matching_pair(self):
        neighbors = {
            1: {"item_ids": [11, 12], "scores": [0.99, 0.98]},
            2: {"item_ids": [21, 22], "scores": [0.40, 0.39]},
            3: {"item_ids": [31, 32], "scores": [0.80, 0.10]},
            4: {"item_ids": [41, 42], "scores": [0.40, 0.39]},
        }
        examples = [
            {"positive_item_ids": [1], "negative_item_ids": [3]},
            {"positive_item_ids": [2], "negative_item_ids": [4]},
        ]

        hardness = compute_sample_hardness(examples, neighbors)

        self.assertGreater(hardness[0], hardness[1])

    def test_multi_item_sample_averages_pair_deltas(self):
        neighbors = {
            1: {"item_ids": [10, 11], "scores": [0.9, 0.1]},
            2: {"item_ids": [10, 11], "scores": [0.6, 0.4]},
            3: {"item_ids": [10, 11], "scores": [0.5, 0.5]},
            4: {"item_ids": [10, 11], "scores": [0.4, 0.6]},
        }
        examples = [
            {"positive_item_ids": [1, 2], "negative_item_ids": [3, 4]},
            {"positive_item_ids": [2], "negative_item_ids": [2]},
        ]

        hardness = compute_sample_hardness(examples, neighbors)

        self.assertEqual(len(hardness), 2)
        self.assertGreater(hardness[0], hardness[1])

    def test_attach_hardness_does_not_mutate_inputs(self):
        examples = [{"positive_item_ids": [1], "negative_item_ids": [2]}]

        attached = attach_hardness(examples, [0.75])

        self.assertNotIn("hardness", examples[0])
        self.assertEqual(attached[0]["hardness"], 0.75)


class ManifestTest(unittest.TestCase):
    def test_changed_input_file_invalidates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps([1]), encoding="utf-8")
            cached = build_manifest(
                dataset="demo",
                model_id="model",
                input_files=[source],
                top_k=10,
                seed=2025,
                missing_images=0,
            )
            source.write_text(json.dumps([1, 2]), encoding="utf-8")
            expected = build_manifest(
                dataset="demo",
                model_id="model",
                input_files=[source],
                top_k=10,
                seed=2025,
                missing_images=0,
            )

            with self.assertRaises(StaleArtifactError):
                validate_manifest(cached, expected)

    def test_changed_encoder_model_invalidates_embedding_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            titles = root / "titles.csv"
            images = root / "images"
            titles.write_text("item,title\n1,A\n", encoding="utf-8")
            images.mkdir()
            cached = build_embedding_manifest(
                dataset="demo", model_id="model-a", title_path=titles, image_dir=images
            )
            expected = build_embedding_manifest(
                dataset="demo", model_id="model-b", title_path=titles, image_dir=images
            )

            with self.assertRaises(StaleArtifactError):
                validate_manifest(cached, expected)

    def test_identical_manifest_is_valid(self):
        manifest = {
            "schema_version": 1,
            "dataset": "demo",
            "model_id": "model",
            "top_k": 10,
            "seed": 2025,
            "missing_images": 0,
            "inputs": [],
        }

        validate_manifest(manifest, dict(manifest))


if __name__ == "__main__":
    unittest.main()
