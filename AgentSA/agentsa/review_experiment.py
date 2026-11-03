from __future__ import annotations

import time

import numpy as np

from .agents import agent_directory
from .attacks import as_array, load_attack
from .evaluation import evaluate_target
from .review_data import load_review_data
from .review_generation import load_generated_reviews
from .review_models import ReviewTextEncoder, aggregate_features, means
from .review_training import fit_review_model
from .utils import digest, file_hash, resolve, write_csv, write_json


def _diagnostics(rows, encoder, target_text):
    fillers = [row for row in rows if row.get("round", -1) >= 0]
    if not fillers:
        return {"changed_review_fraction": 0.0, "mean_shared_features": 0.0,
                "mean_target_similarity_gain": 0.0}
    initial = [row.get("initial_review", row["review"]) for row in fillers]
    final = [row["review"] for row in fillers]
    target = encoder.encode([target_text])[0]
    a, b = encoder.encode(initial), encoder.encode(final)
    return {"changed_review_fraction": float(np.mean([x != y for x, y in zip(initial, final)])),
            "mean_shared_features": float(np.mean([len(row.get("shared_features", [])) for row in fillers])),
            "mean_target_similarity_gain": float(np.mean(b @ target - a @ target))}


def run_review_experiments(config, dataset, models, seeds, targets, mock=False):
    review_data = load_review_data(config, dataset)
    encoder = ReviewTextEncoder(config["reviews"], mock=mock)
    users, items = dataset["manifest"]["users"], dataset["manifest"]["items"]
    base = aggregate_features(review_data["train"], users, items, encoder)
    clean_features = means(base)
    mean = float(dataset["train"][:, 2].mean())
    root = resolve(config, config["reviews"]["experiment_output"]) / ("mock_validation" if mock else "experiments")
    results = []
    for model_name in models:
        for seed in seeds:
            clean = fit_review_model(model_name, dataset["train"], dataset["validation"], users, items,
                clean_features, config, mean, seed, root / "clean" / model_name / str(seed),
                {"dataset_id": dataset["manifest"]["dataset_id"], "mode": "clean",
                 "review_data": review_data["manifest"]["identity"], "encoder": encoder.identity})
            for target in targets:
                clean_metrics, clean_scores, _ = evaluate_target(clean, target, dataset["train"],
                    dataset["test"], users, items, config)
                rating_dir = agent_directory(config, target, mock)
                fake_ratings = load_attack(rating_dir / "fake_ratings.csv")
                rating_hash = file_hash(rating_dir / "fake_ratings.csv")
                for mode in ("rating_only", "plain", "tfp"):
                    fake_reviews = []
                    review_manifest = None
                    if mode != "rating_only":
                        fake_reviews, review_manifest = load_generated_reviews(config, dataset, target, mode, mock)
                        review_pairs = {(r["user_id"], r["item_id"], float(r["rating"])) for r in fake_reviews}
                        rating_pairs = {(r["user_id"], r["item_id"], float(r["rating"])) for r in fake_ratings}
                        if review_pairs != rating_pairs:
                            raise ValueError("Review and rating attacks do not contain identical interactions.")
                    total_users = users + int(read_json_safe(rating_dir / "manifest.json")["fake_users"])
                    aggregates = aggregate_features(fake_reviews, total_users, items, encoder, base=base)
                    text_features = means(aggregates)
                    poisoned_train = np.concatenate([dataset["train"], as_array(fake_ratings)])
                    identity = {"dataset_id": dataset["manifest"]["dataset_id"], "mode": mode,
                                "rating_sha256": rating_hash, "review_signature":
                                review_manifest["signature"] if review_manifest else None,
                                "review_sha256": review_manifest[f"{mode}_sha256"] if review_manifest else None,
                                "encoder": encoder.identity}
                    directory = root / "poison" / model_name / f"target_{target}" / mode / str(seed)
                    started = time.monotonic()
                    poisoned = fit_review_model(model_name, poisoned_train, dataset["validation"],
                        total_users, items, text_features, config, mean, seed, directory, identity)
                    metrics, scores, _ = evaluate_target(poisoned, target, dataset["train"],
                        dataset["test"], users, items, config)
                    target_item = dataset["items"][target]
                    target_text = " ".join(str(target_item.get(k, "")) for k in ("title", "category", "description"))
                    diagnostics = _diagnostics(fake_reviews, encoder, target_text) if fake_reviews else {
                        "changed_review_fraction": 0.0, "mean_shared_features": 0.0,
                        "mean_target_similarity_gain": 0.0}
                    row = {"model": model_name, "target": target, "seed": seed, "review_mode": mode,
                           "mock_validation": mock, "rating_sha256": rating_hash,
                           "eligible_users": metrics["eligible_users"],
                           "signed_prediction_shift": float((scores-clean_scores).mean()),
                           "clean_rmse": clean_metrics["rmse"], "poison_rmse": metrics["rmse"],
                           "clean_mae": clean_metrics["mae"], "poison_mae": metrics["mae"],
                           "elapsed_seconds": time.monotonic()-started, **diagnostics}
                    for k in (10, 50, 100):
                        row[f"clean_hr_at_{k}"] = clean_metrics[f"hr_at_{k}"]
                        row[f"poison_hr_at_{k}"] = metrics[f"hr_at_{k}"]
                        row[f"delta_hr_at_{k}"] = metrics[f"hr_at_{k}"]-clean_metrics[f"hr_at_{k}"]
                    results.append(row)
                    write_json(directory / "metrics.json", row)
                    write_csv(root / "results.csv", results, list(results[0]))
    contrasts = []
    for model_name in models:
        for target in targets:
            for seed in seeds:
                group = {(r["review_mode"]): r for r in results
                         if r["model"] == model_name and r["target"] == target and r["seed"] == seed}
                for left, right, label in (("plain", "rating_only", "plain_minus_rating_only"),
                                           ("tfp", "plain", "tfp_minus_plain")):
                    record = {"model": model_name, "target": target, "seed": seed, "contrast": label}
                    for metric in ("signed_prediction_shift", "poison_rmse", "poison_mae",
                                   "poison_hr_at_10", "poison_hr_at_50", "poison_hr_at_100"):
                        record[metric] = group[left][metric] - group[right][metric]
                    contrasts.append(record)
    write_json(root / "contrasts.json", contrasts)
    write_json(root / "summary.json", {"scope": "review-aware engineering adaptation",
        "causal_contrast": "TFP minus plain uses identical users/items/ratings",
        "rows": len(results), "models": models, "targets": targets, "seeds": seeds})
    return results


def read_json_safe(path):
    from .utils import read_json
    return read_json(path)
