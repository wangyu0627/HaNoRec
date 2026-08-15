from __future__ import annotations

import unittest

from hanorec.data.metadata import extract_hanorec_metadata, pad_item_ids


class MetadataTest(unittest.TestCase):
    def test_extracts_aligned_metadata_without_mutating_features(self) -> None:
        features = [
            {
                "hanorec_hardness": 0.75,
                "hanorec_positive_item_ids": [1],
                "hanorec_negative_item_ids": [9],
            },
            {
                "hanorec_hardness": 1.25,
                "hanorec_positive_item_ids": [2, 3],
                "hanorec_negative_item_ids": [8, 7],
            },
        ]

        metadata = extract_hanorec_metadata(features)

        self.assertEqual(metadata["hardness"], [0.75, 1.25])
        self.assertEqual(metadata["positive_item_ids"], [[1], [2, 3]])
        self.assertEqual(metadata["negative_item_ids"], [[9], [8, 7]])
        self.assertNotIn("hardness", features[0])

    def test_plain_dpo_batch_returns_empty_metadata(self) -> None:
        self.assertEqual(extract_hanorec_metadata([{"input_ids": [1]}]), {})

    def test_partial_metadata_is_rejected(self) -> None:
        features = [
            {
                "hanorec_hardness": 1.0,
                "hanorec_positive_item_ids": [1],
                "hanorec_negative_item_ids": [2],
            },
            {"input_ids": [3]},
        ]
        with self.assertRaisesRegex(ValueError, "every sample"):
            extract_hanorec_metadata(features)

    def test_misaligned_item_pairs_are_rejected(self) -> None:
        features = [
            {
                "hanorec_hardness": 1.0,
                "hanorec_positive_item_ids": [1, 2],
                "hanorec_negative_item_ids": [3],
            }
        ]
        with self.assertRaisesRegex(ValueError, "aligned"):
            extract_hanorec_metadata(features)

    def test_item_ids_are_padded_with_sentinel(self) -> None:
        self.assertEqual(pad_item_ids([[1], [2, 3]]), [[1, -1], [2, 3]])


if __name__ == "__main__":
    unittest.main()
