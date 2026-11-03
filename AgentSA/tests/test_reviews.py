import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agentsa.agents import generate_agent_attack
from agentsa.data import load_dataset, prepare
from agentsa.review_data import load_review_data, prepare_review_data
from agentsa.review_experiment import run_review_experiments
from agentsa.review_generation import (MockReviewClient, generate_review_attack,
                                        load_generated_reviews, valid_propagated)
from agentsa.review_models import ReviewTextEncoder, aggregate_features, build_review_model, means
from agentsa.utils import load_config, read_json
from test_core import ROOT, create_data


def review_config(root):
    config = load_config(ROOT / "configs/review_smoke.yaml")
    config["project_root"] = str(root)
    config["data"].update(reviews="reviews.gz", metadata="meta.gz", output="processed",
                          min_interactions=2, max_users=None, target_count=1)
    config["attack"].update(fake_ratio=.2, filler_count=4)
    config["agent"].update(output="rating_agent", rounds=2, exposure_size=6,
                           max_selected_per_round=2, filler_cap=4, reference_per_round=2,
                           profile_candidate_multiplier=1, validation_retries=2)
    config["reviews"].update(data_output="review_data", agent_output="review_agent",
                             experiment_output="review_experiment")
    return config


class ReviewDataTests(unittest.TestCase):
    def test_review_sidecar_exactly_matches_frozen_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = review_config(root)
            prepare(config)
            dataset = load_dataset(config)
            manifest = prepare_review_data(config, dataset)
            loaded = load_review_data(config, dataset)
            self.assertEqual(sum(manifest["counts"].values()), dataset["manifest"]["interactions"])
            self.assertEqual(manifest["counts"]["train"], len(dataset["train"]))
            self.assertTrue(all(row["review"] for row in loaded["train"]))
            self.assertEqual(prepare_review_data(config, dataset)["identity"], manifest["identity"])

    def test_sidecar_reuse_ignores_unrelated_review_output_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = review_config(root)
            prepare(config)
            dataset = load_dataset(config)
            prepare_review_data(config, dataset)
            config["reviews"]["agent_output"] = "another_review_agent"
            config["reviews"]["experiment_output"] = "another_experiment"
            self.assertEqual(len(load_review_data(config, dataset)["train"]), len(dataset["train"]))


class ReviewGenerationTests(unittest.TestCase):
    def test_plain_and_tfp_share_every_rating_and_resume_without_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = review_config(root)
            prepare(config)
            dataset = load_dataset(config)
            target = dataset["targets"][0]
            generate_agent_attack(config, dataset, target, mock=True)
            path = generate_review_attack(config, dataset, target, mock=True)
            manifest = read_json(path)
            plain, _ = load_generated_reviews(config, dataset, target, "plain", True)
            tfp, _ = load_generated_reviews(config, dataset, target, "tfp", True)
            triples = lambda rows: {(r["user_id"], r["item_id"], float(r["rating"])) for r in rows}
            self.assertEqual(triples(plain), triples(tfp))
            self.assertEqual(len(plain), manifest["plain_interactions"])
            plain_text = {(r["user_id"], r["item_id"]): r["review"] for r in plain}
            for row in tfp:
                if row["round"] >= 0:
                    self.assertEqual(row["initial_review"], plain_text[(row["user_id"], row["item_id"])])
            with patch.object(MockReviewClient, "generate", side_effect=AssertionError("must resume from states")):
                generate_review_attack(config, dataset, target, mock=True)

    def test_propagation_rejects_target_name_and_wrong_ids(self):
        target = {"title": "Secret Cat Brush", "asin": "A1"}
        bad = {"reviews": [{"item_id": 1, "review": "Secret Cat Brush is great",
                             "shared_features": []}]}
        with self.assertRaisesRegex(ValueError, "names the target"):
            valid_propagated(bad, [1], target)
        with self.assertRaisesRegex(ValueError, "IDs"):
            valid_propagated({"reviews": [{"item_id": 2, "review": "Fine",
                                            "shared_features": []}]}, [1], target)


class ReviewModelTests(unittest.TestCase):
    def test_review_models_are_text_sensitive(self):
        settings = load_config(ROOT / "configs/review_smoke.yaml")["reviews"]
        encoder = ReviewTextEncoder(settings, mock=True)
        rows_a = [{"user_id": 0, "item_id": 0, "review": "gentle dental care"}]
        rows_b = [{"user_id": 0, "item_id": 0, "review": "metal outdoor cage"}]
        features_a = means(aggregate_features(rows_a, 2, 2, encoder))
        features_b = means(aggregate_features(rows_b, 2, 2, encoder))
        for name in ("dual_tower_early", "dual_tower_late"):
            model_a = build_review_model(name, 2, 2, features_a, settings["training"], 3., 11)
            model_b = build_review_model(name, 2, 2, features_b, settings["training"], 3., 11)
            with torch.no_grad():
                a = model_a(torch.tensor([0]), torch.tensor([0]))
                b = model_b(torch.tensor([0]), torch.tensor([0]))
            self.assertFalse(torch.equal(a, b), name)

    def test_mock_review_experiment_end_to_end_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = review_config(root)
            prepare(config)
            dataset = load_dataset(config)
            prepare_review_data(config, dataset)
            target = dataset["targets"][0]
            with patch("requests.post", side_effect=AssertionError("No network allowed")):
                generate_agent_attack(config, dataset, target, mock=True)
                generate_review_attack(config, dataset, target, mock=True)
                results = run_review_experiments(config, dataset,
                    ["dual_tower_early", "dual_tower_late"], [11], [target], mock=True)
            self.assertEqual(len(results), 6)
            self.assertEqual({r["review_mode"] for r in results}, {"rating_only", "plain", "tfp"})
            for model in ("dual_tower_early", "dual_tower_late"):
                hashes = {r["rating_sha256"] for r in results if r["model"] == model}
                self.assertEqual(len(hashes), 1)
            tfp = [r for r in results if r["review_mode"] == "tfp"]
            self.assertTrue(all(r["changed_review_fraction"] == 1 for r in tfp))
            self.assertTrue((root / "review_experiment" / "mock_validation" / "results.csv").exists())
            self.assertTrue((root / "review_experiment" / "mock_validation" / "contrasts.json").exists())


if __name__ == "__main__":
    unittest.main()
