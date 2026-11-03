import unittest

import numpy as np
import torch

from agentsa.preflight import compare_ranks, rank_catalog, select_candidates, GRID, SEEDS


class RankingDiagnosticsTests(unittest.TestCase):
    def test_seen_mask_target_exclusion_ties_and_cutoff(self):
        class Scores(torch.nn.Module):
            def forward(self, users, items):
                values = torch.tensor([[9., 4., 4., 3., 2.], [0., 4., 4., 3., 2.]])
                return values[users, items]
        dataset = {"manifest": {"users": 2, "items": 5},
                   "train": np.array([[0, 0, 5], [1, 2, 4]])}
        config = {"training": {"device": "cpu"}, "experiment": {
            "ranking_user_batch": 2, "prediction_batch": 3, "clip_predictions": False}}
        result = rank_catalog(Scores(), dataset, config, 2, (1, 2, 3))
        # User 1 saw target and is excluded. User 0's seen 9-point item is masked;
        # item 1 wins the 4-point tie by smaller ID. Zero gap alone is not a hit.
        np.testing.assert_array_equal(result["users"], [0])
        np.testing.assert_array_equal(result["ranks"], [2])
        self.assertEqual(result["gaps"][1][0], 0)
        self.assertEqual(result["gaps"][3][0], 1)

    def test_paired_rank_improvement_and_population_mismatch(self):
        clean = {"users": np.arange(3), "ranks": np.array([100, 10, 5]), "scores": np.zeros(3)}
        poison = {"users": np.arange(3), "ranks": np.array([50, 10, 20]), "scores": np.ones(3)}
        result = compare_ranks(clean, poison)
        self.assertAlmostEqual(result["users_improved_fraction"], 1/3)
        self.assertAlmostEqual(result["users_worsened_fraction"], 1/3)
        self.assertAlmostEqual(result["rank_improvement_mean"], 35/3)
        poison["users"] = np.array([0, 1, 4])
        with self.assertRaises(ValueError):
            compare_ranks(clean, poison)

    def test_selection_uses_validation_and_requires_complete_seeds(self):
        rows = []
        for model, rates in GRID.items():
            for rate in rates:
                for seed in SEEDS:
                    rows.append({"model": model, "learning_rate": rate, "seed": seed,
                        "best_validation_rmse": 1 + rate, "best_epoch": 4, "epochs_run": 14,
                        "test_rmse": 100 - rate})
        _, chosen = select_candidates(rows)
        self.assertEqual(chosen["nmf"]["learning_rate"], 0.0003)
        self.assertEqual(chosen["neunmf"]["learning_rate"], 0.0001)
        with self.assertRaises(ValueError):
            select_candidates(rows[:-1])


if __name__ == "__main__":
    unittest.main()
