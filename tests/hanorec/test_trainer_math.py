from __future__ import annotations

import math
import unittest

from hanorec.trainer.math import hanorec_dpo_terms


class HaNoRecTrainerMathTest(unittest.TestCase):
    def test_constant_reward_gap_keeps_base_beta_for_unit_hardness(self) -> None:
        terms = hanorec_dpo_terms(
            policy_chosen=[-1.0, -2.0],
            policy_rejected=[-2.0, -3.0],
            reference_chosen=[-1.5, -2.5],
            reference_rejected=[-2.5, -3.5],
            hardness=[1.0, 1.0],
            beta0=0.1,
        )

        self.assertAlmostEqual(terms["responsiveness"], 1.0)
        self.assertEqual(terms["betas"], [0.1, 0.1])

    def test_higher_hardness_increases_beta_and_separates_rewards(self) -> None:
        terms = hanorec_dpo_terms(
            policy_chosen=[-1.0, -1.0],
            policy_rejected=[-2.0, -2.0],
            reference_chosen=[-1.5, -1.5],
            reference_rejected=[-2.0, -2.0],
            hardness=[0.5, 1.5],
            beta0=0.1,
        )

        self.assertLess(terms["betas"][0], terms["betas"][1])
        self.assertGreater(terms["reward_margins"][1], terms["reward_margins"][0])
        self.assertGreater(terms["losses"][0], terms["losses"][1])

    def test_loss_matches_negative_log_sigmoid(self) -> None:
        terms = hanorec_dpo_terms(
            policy_chosen=[-1.0],
            policy_rejected=[-2.0],
            reference_chosen=[-1.5],
            reference_rejected=[-2.0],
            hardness=[1.0],
            beta0=0.2,
        )
        expected = math.log1p(math.exp(-0.2 * 0.5))
        self.assertAlmostEqual(terms["losses"][0], expected)

    def test_rejects_misaligned_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            hanorec_dpo_terms(
                policy_chosen=[-1.0],
                policy_rejected=[-2.0, -3.0],
                reference_chosen=[-1.5],
                reference_rejected=[-2.0],
                hardness=[1.0],
                beta0=0.1,
            )


if __name__ == "__main__":
    unittest.main()
