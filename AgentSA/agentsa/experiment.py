from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from .agents import agent_directory
from .attacks import (METHODS, as_array, fake_user_count, load_attack, save_attack,
                      traditional_attack, validate_fake)
from .evaluation import evaluate_target
from .training import fit_model
from .utils import (digest, file_hash, read_json, resolve, stable_seed,
                    write_csv, write_json)


def root_output(config, mock=False):
    return resolve(config, config["experiment"]["output"]) / ("mock_validation" if mock else "experiments")


def train_clean(config, dataset, models, seeds, mock=False):
    manifest = dataset["manifest"]
    root = root_output(config, mock)
    mean = float(dataset["train"][:, 2].mean())
    for name in models:
        for seed in seeds:
            print(f"Clean {name}, seed {seed}", flush=True)
            fit_model(name, dataset["train"], dataset["validation"], manifest["users"],
                      manifest["items"], config, mean, seed, root / "clean" / name / str(seed),
                      {"dataset_id": manifest["dataset_id"], "mode": "clean"})


def checked_agent_data(config, dataset, target, mock):
    directory = agent_directory(config, target, mock)
    manifest = read_json(directory / "manifest.json")
    if manifest["dataset_id"] != dataset["manifest"]["dataset_id"] or manifest["target"] != target:
        raise ValueError("Agent data belongs to another dataset/target.")
    if manifest["mock"] != mock:
        raise ValueError("Mock and real Agent data cannot be mixed.")
    path = directory / "fake_ratings.csv"
    if file_hash(path) != manifest["sha256"]:
        raise ValueError("Agent CSV hash mismatch.")
    count = fake_user_count(dataset["manifest"]["users"], config["attack"])
    if manifest["fake_users"] != count:
        raise ValueError("Agent fake-user count differs from experiment settings.")
    rows = load_attack(path)
    if manifest.get("variant", "rating_original") != config["agent"].get("variant", "rating_original"):
        raise ValueError("Agent variant differs from experiment config; choose matching generated data.")
    if "generation_settings" in manifest and manifest["generation_settings"] != config["agent"]:
        raise ValueError("Agent generation settings differ from experiment config.")
    validate_fake(rows, dataset["manifest"]["users"], dataset["manifest"]["items"], target, count)
    lengths = [sum(r["user_id"] == dataset["manifest"]["users"] + a and r["item_id"] != target for r in rows)
               for a in range(count)]
    if lengths != manifest["filler_lengths"]:
        raise ValueError("Agent manifest filler counts do not match its data.")
    return rows, lengths, manifest


def run_experiments(config, dataset, models, methods, seeds, targets,
                    include_agent=False, match_agent=False, mock=False):
    if set(methods) - METHODS:
        raise ValueError(f"Unknown attacks: {set(methods) - METHODS}")
    if match_agent and not include_agent:
        raise ValueError("--match-agent-budget requires --include-agent.")
    manifest = dataset["manifest"]
    users, items = manifest["users"], manifest["items"]
    count = fake_user_count(users, config["attack"])
    root = root_output(config, mock)
    root.mkdir(parents=True, exist_ok=True)
    invocation = {"config": config, "dataset_id": manifest["dataset_id"], "models": models,
                  "methods": methods, "seeds": seeds, "targets": targets, "mock": mock,
                  "include_agent": include_agent, "match_agent_budget": match_agent,
                  "evaluation_sha256": file_hash(Path(__file__).with_name("evaluation.py")),
                  "experiment_sha256": file_hash(Path(__file__))}
    write_json(root / f"invocation_{digest(invocation)[:16]}.json", invocation)
    mean = float(dataset["train"][:, 2].mean())
    attack_sets = {}
    for target in targets:
        agent_rows, lengths, agent_manifest = None, [config["attack"]["filler_count"]] * count, None
        if include_agent:
            agent_rows, agent_lengths, agent_manifest = checked_agent_data(config, dataset, target, mock)
            if match_agent:
                lengths = agent_lengths
        for method in methods:
            rows = traditional_attack(method, dataset["train"], users, items, target, count, lengths,
                                      config["attack"], stable_seed(config["attack"]["seed"], target, method))
            attack_sets[(target, method)] = rows
        if agent_rows is not None:
            attack_sets[(target, "agent")] = agent_rows
        for method in [*methods, *(["agent"] if include_agent else [])]:
            rows = attack_sets[(target, method)]
            attack_id = digest({"rows": rows, "dataset": manifest["dataset_id"], "target": target})
            path = root / "attacks" / f"target_{target}" / f"{method}_{attack_id[:12]}.csv"
            save_attack(path, rows)
            write_json(path.with_suffix(".json"), {
                "attack_id": attack_id, "dataset_id": manifest["dataset_id"], "target": target,
                "method": method, "fake_users": count, "interactions": len(rows), "mock": mock,
                "budget_protocol": "matched_agent" if match_agent else "fixed_baselines_natural_agent",
                "agent_manifest": agent_manifest if method == "agent" else None,
            })
    results = []
    for name in models:
        for seed in seeds:
            clean = fit_model(name, dataset["train"], dataset["validation"], users, items,
                              config, mean, seed, root / "clean" / name / str(seed),
                              {"dataset_id": manifest["dataset_id"], "mode": "clean"})
            for target in targets:
                clean_metrics, clean_scores, _ = evaluate_target(clean, target, dataset["train"],
                    dataset["test"], users, items, config)
                for method in [*methods, *(["agent"] if include_agent else [])]:
                    rows = attack_sets[(target, method)]
                    attack_id = digest({"rows": rows, "dataset": manifest["dataset_id"], "target": target})
                    run = root / "poison" / name / f"target_{target}" / method / attack_id[:12] / str(seed)
                    print(f"{'MOCK validation' if mock else 'Experiment'}: {name}, target {target}, {method}, seed {seed}", flush=True)
                    started = time.monotonic()
                    poisoned_train = np.concatenate([dataset["train"], as_array(rows)], axis=0)
                    poisoned = fit_model(name, poisoned_train, dataset["validation"], users + count, items,
                                         config, mean, seed, run,
                                         {"dataset_id": manifest["dataset_id"], "mode": "poison", "attack_id": attack_id})
                    poison_metrics, poison_scores, _ = evaluate_target(poisoned, target, dataset["train"],
                        dataset["test"], users, items, config)
                    difference = poison_scores - clean_scores
                    result = {
                        "model": name, "target": target, "seed": seed, "method": method,
                        "mock_validation": mock, "dataset_id": manifest["dataset_id"], "attack_id": attack_id,
                        "budget_protocol": "matched_agent" if match_agent else "fixed_baselines_natural_agent",
                        "top_k": config["experiment"]["top_k"], "eligible_users": clean_metrics["eligible_users"],
                        "clean_hr": clean_metrics["hr"], "poison_hr": poison_metrics["hr"],
                        "delta_hr": poison_metrics["hr"] - clean_metrics["hr"],
                        "signed_prediction_shift": float(difference.mean()),
                        "absolute_prediction_shift": float(np.abs(difference).mean()),
                        "clean_rmse": clean_metrics["rmse"], "poison_rmse": poison_metrics["rmse"],
                        "delta_rmse": poison_metrics["rmse"] - clean_metrics["rmse"],
                        "clean_mae": clean_metrics["mae"], "poison_mae": poison_metrics["mae"],
                        "delta_mae": poison_metrics["mae"] - clean_metrics["mae"],
                        "fake_users": count, "fake_ratings": len(rows), "target_ratings": count,
                        "filler_ratings": len(rows) - count,
                        "fake_ratio": config["attack"]["fake_ratio"],
                        "agent_variant": config["agent"].get("variant", "rating_original") if include_agent else "not_included",
                        "mean_rank_improvement": clean_metrics["mean_target_rank"] - poison_metrics["mean_target_rank"],
                        "elapsed_train_and_evaluation_seconds": time.monotonic() - started,
                    }
                    for k in (10, 50, 100):
                        result[f"clean_hr_at_{k}"] = clean_metrics[f"hr_at_{k}"]
                        result[f"poison_hr_at_{k}"] = poison_metrics[f"hr_at_{k}"]
                        result[f"delta_hr_at_{k}"] = poison_metrics[f"hr_at_{k}"] - clean_metrics[f"hr_at_{k}"]
                    write_json(run / "metrics.json", result)
                    results.append(result)
                    write_csv(root / "results.csv", results, list(results[0]))
    summary = summarize(results)
    write_json(root / "summary.json", summary)
    write_json(root / "target_coverage.json", target_coverage(results,
        config["experiment"].get("target_success_hr", 0.01)))
    print(f"Results: {root / 'results.csv'}")
    return results


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[(row["model"], row["method"], row["mock_validation"], row["budget_protocol"])].append(row)
    output = []
    for (model, method, mock, budget), rows in sorted(groups.items()):
        summary = {"model": model, "method": method, "mock_validation": mock, "budget_protocol": budget,
                   "n_target_seed_pairs": len(rows), "targets": len({r["target"] for r in rows}),
                   "training_seeds": len({r["seed"] for r in rows}),
                   "note": "Descriptive mean/std over target-seed pairs; not independent generation replicates."}
        for metric in ("clean_hr", "poison_hr", "delta_hr", "signed_prediction_shift", "absolute_prediction_shift",
                       "delta_rmse", "delta_mae", "fake_ratings",
                       *[f"{prefix}_hr_at_{k}" for k in (10, 50, 100) for prefix in ("clean", "poison", "delta")]):
            values = [r[metric] for r in rows]
            summary[metric + "_mean"] = float(np.mean(values))
            summary[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def target_coverage(results, minimum_hr=0.01):
    """Target coverage across the evaluated target set, after averaging training seeds.

    Any-hit coverage is distinct from reaching a preset minimum exposure rate.
    Target IDs and failures are retained, including zero-HR targets.
    """
    groups = defaultdict(list)
    for row in results:
        key = (row["model"], row["method"], row["mock_validation"], row["budget_protocol"],
               row.get("agent_variant", "rating_original"))
        groups[key].append(row)
    output = []
    for (model, method, mock, budget, variant), rows in sorted(groups.items()):
        targets = sorted({r["target"] for r in rows})
        record = {"model": model, "method": method, "mock_validation": mock,
                  "budget_protocol": budget, "agent_variant": variant, "targets": len(targets),
                  "minimum_hr": minimum_hr, "scope": "evaluated frozen targets only; mean HR across training seeds",
                  "per_target": []}
        for target in targets:
            target_rows = [r for r in rows if r["target"] == target]
            if len({r["seed"] for r in target_rows}) != len(target_rows):
                raise ValueError("Duplicate target/training seed rows in coverage report.")
            item = {"target": target, "seeds": sorted(r["seed"] for r in target_rows)}
            for k in (10, 50, 100):
                item[f"clean_hr_at_{k}"] = float(np.mean([r[f"clean_hr_at_{k}"] for r in target_rows]))
                item[f"poison_hr_at_{k}"] = float(np.mean([r[f"poison_hr_at_{k}"] for r in target_rows]))
            record["per_target"].append(item)
        for k in (10, 50, 100):
            values = [t[f"poison_hr_at_{k}"] for t in record["per_target"]]
            record[f"fraction_targets_any_hit_at_{k}"] = float(np.mean(np.asarray(values) > 0))
            record[f"fraction_targets_minimum_hr_at_{k}"] = float(np.mean(np.asarray(values) >= minimum_hr))
        output.append(record)
    return output
