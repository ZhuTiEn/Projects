import copy
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agentsa.agents import (ExposureEnvironment, FakeUserAgent, generate_agent_attack,
                            valid_ids, valid_ratings)
from agentsa.attacks import (METHODS, as_array, statistics, traditional_attack,
                             validate_fake)
from agentsa.data import load_dataset, prepare, records, split_rows
from agentsa.evaluation import evaluate_target, target_rank_from_scores
from agentsa.experiment import checked_agent_data
from agentsa.llm import BudgetExceeded, ChatClient, MockClient
from agentsa.models import build_model
from agentsa.training import fit_model, predict
from agentsa.utils import load_config, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_config(root):
    config = load_config(ROOT / "configs/smoke.yaml")
    config["project_root"] = str(root)
    config["data"].update(reviews="reviews.gz", metadata="meta.gz", output="processed",
                           min_interactions=2, max_users=None, target_count=1)
    config["attack"].update(fake_ratio=0.2, filler_count=4)
    return config


def create_data(root):
    rows = []
    for user in range(20):
        for offset in range(6):
            item = (user + offset) % 12
            rows.append({"reviewerID": f"U{user:03}", "asin": f"I{item:03}",
                         "overall": 1 + ((user + item) % 5),
                         "summary": f"Summary {user}-{item}",
                         "reviewText": f"User {user} review of pet item {item}",
                         "unixReviewTime": 1388534400 + user * 100 + offset})
    with gzip.open(root / "reviews.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with gzip.open(root / "meta.gz", "wt", encoding="utf-8") as handle:
        for item in range(12):
            # Historical metadata syntax; includes apostrophes and nested arrays.
            handle.write(repr({"asin": f"I{item:03}", "title": f"Dog's toy {item}",
                               "categories": [["Pet Supplies", "Dogs"]]}) + "\n")


class DataTests(unittest.TestCase):
    def test_prepare_coverage_hashes_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = test_config(root)
            manifest = prepare(config)
            dataset = load_dataset(config)
            parts = [set(map(tuple, dataset[name][:, :2].astype(int)))
                     for name in ("train", "validation", "test")]
            self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
            for name in ("validation", "test"):
                for column in (0, 1):
                    self.assertTrue(set(dataset[name][:, column]) <= set(dataset["train"][:, column]))
            self.assertEqual(manifest["interactions"], 120)
            self.assertEqual(prepare(config)["dataset_id"], manifest["dataset_id"])
            path = root / "processed" / "train.csv"
            path.write_text(path.read_text() + "0,0,5,1\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_dataset(config)

    def test_untrusted_literal_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "malicious.txt"
            path.write_text("__import__('os').system('never_execute_this')\n")
            with self.assertRaises(ValueError):
                list(records(path))


class AttackAndEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train = np.asarray([[u, i, 1 + (u + i) % 5]
                                 for u in range(8) for i in range(12)], dtype=float)
        self.settings = {"selected_fraction": 0.25, "aop_fraction": 0.5}

    def test_all_baseline_budgets_and_rating_bounds(self):
        for method in sorted(METHODS):
            with self.subTest(method=method):
                rows = traditional_attack(method, self.train, 8, 12, 0, 4, [2, 3, 4, 5], self.settings, 42)
                validate_fake(rows, 8, 12, 0, 4)
                expected = 4 if method == "target_only" else 4 + 2 + 3 + 4 + 5
                self.assertEqual(len(rows), expected)
                self.assertEqual(rows, traditional_attack(method, self.train, 8, 12, 0, 4,
                                                          [2, 3, 4, 5], self.settings, 42))

    def test_segmented_uses_extreme_ratings(self):
        rows = traditional_attack("segmented", self.train, 8, 12, 0, 1, [4], self.settings, 42)
        self.assertEqual(sorted(r["rating"] for r in rows), [1, 1, 1, 5, 5])

    def test_full_catalog_rank_seen_mask_and_ties(self):
        scores = [5, 4, 4, 3, 5]
        self.assertEqual(target_rank_from_scores(scores, 2, {0, 4}), 2)
        self.assertEqual(target_rank_from_scores(scores, 2, {0, 1, 4}), 1)
        with self.assertRaises(ValueError):
            target_rank_from_scores(scores, 2, {2})

    def test_evaluator_uses_fixed_real_user_population_and_full_catalog(self):
        class FixedModel(torch.nn.Module):
            def forward(self, users, items):
                return items.float()
        config = load_config(ROOT / "configs/smoke.yaml")
        config["experiment"]["top_k"] = 1
        train = np.asarray([[0, 1, 5], [1, 3, 4], [1, 0, 2], [2, 2, 3], [2, 3, 4]], dtype=float)
        test = np.asarray([[1, 2, 2], [2, 0, 0]], dtype=float)
        metrics, scores, ranks = evaluate_target(FixedModel(), 1, train, test, 3, 4, config)
        self.assertEqual(metrics["eligible_users"], 2)
        self.assertEqual(metrics["hr"], 0.5)
        self.assertEqual(metrics["rmse"], 0)
        np.testing.assert_array_equal(ranks, [2, 1])
        np.testing.assert_array_equal(scores, [1, 1])

    def test_schema_rejects_hallucinated_ids_and_duplicate_ratings(self):
        with self.assertRaises(ValueError):
            valid_ids({"item_ids": [99]}, "item_ids", [1, 2])
        with self.assertRaises(ValueError):
            valid_ratings({"ratings": [{"item_id": 1, "rating": 4}, {"item_id": 1, "rating": 5}]}, [1])
        with self.assertRaises(ValueError):
            valid_ratings({"ratings": [{"item_id": 1, "rating": True}]}, [1])

    def test_exposure_excludes_target_and_seen(self):
        env = ExposureEnvironment(np.asarray([10, 5, 1, 0]), 1)
        values = env.expose({1}, 0, 20, 9)
        self.assertEqual(values, [2])


class ModelTests(unittest.TestCase):
    def test_added_users_preserve_real_and_item_initialization(self):
        config = load_config(ROOT / "configs/smoke.yaml")["training"]
        for name in ("nmf", "neunmf"):
            small = build_model(name, 20, 12, config, 3, 42)
            large = build_model(name, 24, 12, config, 3, 42)
            self.assertTrue(torch.equal(small.user.weight, large.user.weight[:20]))
            self.assertTrue(torch.equal(small.item.weight, large.item.weight))
            if name == "neunmf":
                self.assertEqual(small.network[0].in_features,
                                 2 * config["neunmf_dimensions"] + config["neunmf_product_dimensions"])

    def test_training_improves_and_reloads_exact_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp))
            config["training"].update(epochs=35, patience=35, learning_rate=0.02)
            # Fully observed rank-one matrix: meaningful known learning target.
            rows = np.asarray([[u, i, (1.0 + u / 10) * (1.0 + i / 10)]
                               for u in range(8) for i in range(6)], dtype=float)
            for name in ("nmf", "neunmf"):
                with self.subTest(model=name):
                    raw = build_model(name, 8, 6, config["training"], float(rows[:, 2].mean()), 11)
                    before = np.mean((predict(raw, rows) - rows[:, 2]) ** 2)
                    model = fit_model(name, rows, rows, 8, 6, config, float(rows[:, 2].mean()), 11,
                                      Path(tmp) / name, {"test": True})
                    after = np.mean((predict(model, rows) - rows[:, 2]) ** 2)
                    self.assertLess(after, before)
                    if name == "nmf":
                        self.assertTrue(torch.all(model.user.weight >= 0))
                        self.assertTrue(torch.all(model.item.weight >= 0))
                    loaded = fit_model(name, rows, rows, 8, 6, config, float(rows[:, 2].mean()), 11,
                                       Path(tmp) / name, {"test": True})
                    np.testing.assert_array_equal(predict(model, rows), predict(loaded, rows))


class AgentAndTransportTests(unittest.TestCase):
    def test_interrupted_generation_resumes_to_identical_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = test_config(root)
            prepare(config)
            dataset = load_dataset(config)
            target = dataset["targets"][0]
            original_generate = MockClient.generate
            def interrupted(client, task, instruction, payload):
                if task == "select" and payload["agent_id"] == dataset["manifest"]["users"] + 1:
                    raise RuntimeError("Simulated interruption after first user")
                return original_generate(client, task, instruction, payload)
            with patch.object(MockClient, "generate", interrupted):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    generate_agent_attack(config, dataset, target, mock=True)
            resumed_path = generate_agent_attack(config, dataset, target, mock=True)
            resumed = resumed_path.read_bytes()
            config["agent"]["output"] = "fresh_comparison"
            fresh_path = generate_agent_attack(config, dataset, target, mock=True)
            self.assertEqual(resumed, fresh_path.read_bytes())

    def test_mock_resume_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_data(root)
            config = test_config(root)
            prepare(config)
            dataset = load_dataset(config)
            target = dataset["targets"][0]
            path = generate_agent_attack(config, dataset, target, mock=True)
            original = path.read_bytes()
            with patch.object(MockClient, "generate", side_effect=AssertionError("Resume must reuse state")):
                generate_agent_attack(config, dataset, target, mock=True)
            self.assertEqual(original, path.read_bytes())
            rows, lengths, manifest = checked_agent_data(config, dataset, target, True)
            self.assertEqual(sum(lengths), len(rows) - manifest["fake_users"])
            path.write_text(path.read_text() + "999,0,5\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                checked_agent_data(config, dataset, target, True)

    def test_hybrid_memory_union(self):
        class Encoder:
            def relevant(self, item, memory, count):
                return [0, 2][:count]
        memory = [{"item_id": i, "rating": 4} for i in range(5)]
        settings = {"ablation": "full", "relevant_k": 2, "recent_m": 2}
        agent = FakeUserAgent(1, {}, memory, MockClient(), Encoder(), settings)
        self.assertEqual([x["item_id"] for x in agent.retrieve(9)], [0, 2, 3, 4])
        agent.settings = {**settings, "ablation": "no_memory"}
        self.assertEqual(agent.retrieve(9), [])

    def test_http_cache_budget_and_secret_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "AGENTSA_BASE_URL": "http://localhost:9999/v1", "AGENTSA_MODEL": "test-model",
                "AGENTSA_API_KEY": "do-not-persist-this-secret"}):
            settings = load_config(ROOT / "configs/phase1.yaml")["llm"]
            settings["max_requests"] = 1
            client = ChatClient(settings, tmp)
            class Response:
                status_code = 200
                def json(self):
                    return {"choices": [{"message": {"content": '{"ok":true}'}}],
                            "model": "test-model", "usage": {"total_tokens": 12}}
            with patch("agentsa.llm.requests.post", return_value=Response()) as post:
                self.assertEqual(client.generate("task", "instruction", {"x": 1}), {"ok": True})
                self.assertEqual(client.generate("task", "instruction", {"x": 1}), {"ok": True})
                self.assertEqual(post.call_count, 1)
                self.assertTrue(post.call_args.kwargs["headers"]["Authorization"].startswith("Bearer "))
                with self.assertRaises(BudgetExceeded):
                    client.generate("task", "instruction", {"x": 2})
            for path in Path(tmp).rglob("*.json"):
                self.assertNotIn("do-not-persist-this-secret", path.read_text())


if __name__ == "__main__":
    unittest.main()
