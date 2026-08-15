import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from hanorec.data.preference import (
    TitleIndex,
    build_preference_file,
    derive_auc_example,
    derive_ranking_example,
)


def _ranking_sample():
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "The user has watched:\n<image> \"History A\".\n\n"
                    "Below are the candidate movies:\n"
                    "\"Negative A\"\n\"Positive C\"\n\"Negative B\"\n"
                    "\"Positive A\"\n\"Negative C\"\n\"Positive B\"."
                ),
            },
            {
                "role": "assistant",
                "content": '"Positive A"\n"Positive B"\n"Positive C"',
            },
        ],
        "images": ["history-a.jpg"],
    }


class TitleIndexTest(unittest.TestCase):
    def test_duplicate_titles_resolve_to_smallest_item_id(self):
        index = TitleIndex([(9, "Same"), (3, "Same"), (5, "Other")])

        self.assertEqual(index.resolve("Same"), 3)
        self.assertEqual(index.duplicates, {"Same": [3, 9]})

    def test_unknown_title_is_rejected(self):
        index = TitleIndex([(1, "Known")])

        with self.assertRaisesRegex(ValueError, "Unknown title"):
            index.resolve("Missing")


    def test_exact_title_ending_in_period_is_not_cleaned_twice(self):
        index = TitleIndex([(1, "A complete sentence.")])
        self.assertEqual(index.resolve("A complete sentence."), 1)

    def test_duplicate_resolution_can_exclude_known_positive_ids(self):
        index = TitleIndex([(1, "Same"), (2, "Same"), (3, "Other")])
        self.assertEqual(index.resolve("Same", exclude={1}), 2)
        self.assertEqual(index.resolve("Same", exclude={2}), 1)

    def test_structural_quotes_preserve_a_real_terminal_quote(self):
        title = 'Sting: Inside the Songs of "Sacred Love"'
        index = TitleIndex([(1, title)])
        self.assertEqual(index.resolve(f'"{title}"'), 1)

    def test_pandas_none_title_matches_cached_nan_text(self):
        index = TitleIndex([(1, "None")])
        self.assertEqual(index.resolve("nan"), 1)

class PreferenceDerivationTest(unittest.TestCase):
    def setUp(self):
        self.index = TitleIndex(
            [
                (1, "Positive A"),
                (2, "Positive B"),
                (3, "Positive C"),
                (4, "Negative A"),
                (5, "Negative B"),
                (6, "Negative C"),
                (7, "Negative D"),
                (8, "Target Positive"),
            ]
        )

    def test_ranking_derivation_preserves_prompt_and_images(self):
        sample = _ranking_sample()

        result = derive_ranking_example(
            sample,
            self.index,
            positive_item_ids=[1, 2, 3],
            rng=random.Random(2025),
        )

        self.assertEqual(result["messages"], sample["messages"][:1])
        self.assertEqual(result["images"], sample["images"])
        self.assertEqual(result["positive_item_ids"], [1, 2, 3])
        self.assertEqual(len(result["negative_item_ids"]), 3)
        self.assertTrue(set(result["negative_item_ids"]).isdisjoint({1, 2, 3}))
        self.assertEqual(result["chosen"], sample["messages"][1])
        self.assertEqual(result["rejected"]["role"], "assistant")

    def test_ranking_derivation_removes_generated_multiline_outer_quote(self):
        sample = _ranking_sample()
        sample["messages"][1]["content"] = f'"{sample["messages"][1]["content"]}"'

        result = derive_ranking_example(
            sample,
            self.index,
            positive_item_ids=[1, 2, 3],
            rng=random.Random(2025),
        )
        self.assertEqual(result["positive_item_ids"], [1, 2, 3])
    def test_auc_yes_uses_correct_and_opposite_answers(self):
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "The movies the user has not watched:\n\"Negative A\"\n"
                        "Whether the user will like the target movie:\n\"Target Positive\""
                    ),
                },
                {"role": "assistant", "content": "Yes"},
            ],
            "images": ["history.jpg"],
        }

        result = derive_auc_example(sample, self.index, true_item_id=8, rng=random.Random(1))

        self.assertEqual(result["chosen"]["content"], "Yes")
        self.assertEqual(result["rejected"]["content"], "No")
        self.assertEqual(result["positive_item_ids"], [8])
        self.assertEqual(result["negative_item_ids"], [4])

    def test_auc_no_uses_presented_target_as_negative_anchor(self):
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "The movies the user has not watched:\n\"Negative B\"\n"
                        "Whether the user will like the target movie:\n\"Negative C\""
                    ),
                },
                {"role": "assistant", "content": "No"},
            ],
            "images": [],
        }

        result = derive_auc_example(sample, self.index, true_item_id=8, rng=random.Random(1))

        self.assertEqual(result["positive_item_ids"], [8])
        self.assertEqual(result["negative_item_ids"], [6])
        self.assertEqual(result["chosen"]["content"], "No")
        self.assertEqual(result["rejected"]["content"], "Yes")

    def test_file_builder_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.json"
            titles = root / "titles.csv"
            tsv = root / "train.tsv"
            output = root / "derived.json"
            source.write_text(json.dumps([_ranking_sample()]), encoding="utf-8")
            with titles.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["item", "title"])
                writer.writerows([(i, title) for i, title in self.index.items()])
            tsv.write_text("10\t99 98 97 3 2 1\n", encoding="utf-8")

            report = build_preference_file(source, titles, tsv, output, hit=3, seed=2025)

            self.assertEqual(report["samples"], 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["source_index"], 0)
            with self.assertRaises(FileExistsError):
                build_preference_file(source, titles, tsv, output, hit=3, seed=2025)


    def test_file_builder_rebases_existing_images_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            local_image = image_root / "history-a.jpg"
            local_image.write_bytes(b"image")
            sample = _ranking_sample()
            sample["images"] = ["/data/wy/MLLMRec/data/demo/images/history-a.jpg"]
            source = root / "train.json"
            original = json.dumps([sample])
            source.write_text(original, encoding="utf-8")
            titles = root / "titles.csv"
            with titles.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["item", "title"])
                writer.writerows([(i, title) for i, title in self.index.items()])
            tsv = root / "train.tsv"
            tsv.write_text("10\t99 98 97 3 2 1\n", encoding="utf-8")
            output = root / "derived.json"

            report = build_preference_file(source, titles, tsv, output, hit=3)
            derived = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(Path(derived[0]["images"][0]), local_image.resolve())
            self.assertEqual(source.read_text(encoding="utf-8"), original)
            self.assertEqual(report["rebased_images"], 1)

if __name__ == "__main__":
    unittest.main()
