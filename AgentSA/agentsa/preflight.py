"""Offline clean-model selection and diagnostics of completed pilot checkpoints.

No LLM client is imported or called. Original pilot files are read-only inputs.
Run: python -m agentsa.preflight --help
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .data import load_dataset
from .evaluation import prediction_metrics, target_rank_from_scores, train_histories
from .models import build_model
from .training import fit_model, predict
from .utils import (digest, file_hash, load_config, read_csv, read_json,
                    resolve, seed_everything, write_csv, write_json)


GRID = {"nmf": [0.001, 0.0003], "neunmf": [0.001, 0.0003, 0.0001]}
SEEDS = [11, 22, 33]


def summarize_ranks(ranks, scores, gaps, cutoffs=(10, 50, 100)):
    result = {"users": len(ranks), "mean_target_score": float(np.mean(scores))}
    for q in (10, 25, 50, 75, 90):
        result[f"rank_p{q}"] = float(np.percentile(ranks, q))
    result["rank_mean"] = float(np.mean(ranks))
    for k in cutoffs:
        result[f"hr_at_{k}"] = float(np.mean(ranks <= k))
        result[f"score_gap_top_{k}_median"] = float(np.median(gaps[k]))
    return result


def rank_catalog(model, dataset, config, target, cutoffs=(10, 50, 100)):
    """Use the original full-catalog population, seen mask, and ID tie rule.

    gap = target score - kth highest eligible score, including target. A zero
    gap can be a score tie; only rank (with ID tie break) determines a hit.
    """
    users, items = dataset["manifest"]["users"], dataset["manifest"]["items"]
    histories = train_histories(dataset["train"], users)
    eligible = np.asarray([u for u in range(users) if target not in histories[u]], dtype=np.int64)
    settings = config["experiment"]
    ids = np.arange(items)
    ranks, scores = [], []
    gaps = {k: [] for k in cutoffs}
    for start in range(0, len(eligible), settings["ranking_user_batch"]):
        batch_users = eligible[start:start + settings["ranking_user_batch"]]
        pairs = np.column_stack([np.repeat(batch_users, items), np.tile(ids, len(batch_users))])
        values = predict(model, pairs, config["training"]["device"],
                         settings["prediction_batch"], settings["clip_predictions"]).reshape(-1, items)
        for user, value in zip(batch_users, values):
            seen = histories[user]
            ranks.append(target_rank_from_scores(value, target, seen))
            scores.append(float(value[target]))
            mask = np.ones(items, dtype=bool)
            mask[list(seen)] = False
            candidate_scores = value[mask]
            if max(cutoffs) > len(candidate_scores):
                raise ValueError("Requested ranking cutoff exceeds the unseen catalog.")
            positions = [len(candidate_scores) - k for k in cutoffs]
            partitioned = np.partition(candidate_scores, positions)
            for k, position in zip(cutoffs, positions):
                gaps[k].append(float(value[target] - partitioned[position]))
    return {"users": eligible, "ranks": np.asarray(ranks), "scores": np.asarray(scores),
            "gaps": {k: np.asarray(value) for k, value in gaps.items()}}


def compare_ranks(clean, poisoned):
    if not np.array_equal(clean["users"], poisoned["users"]):
        raise ValueError("Clean and poison populations differ.")
    improvement = clean["ranks"] - poisoned["ranks"]
    return {"rank_improvement_mean": float(improvement.mean()),
            "rank_improvement_median": float(np.median(improvement)),
            "users_improved_fraction": float(np.mean(improvement > 0)),
            "users_unchanged_fraction": float(np.mean(improvement == 0)),
            "users_worsened_fraction": float(np.mean(improvement < 0)),
            "signed_prediction_shift": float(np.mean(poisoned["scores"] - clean["scores"]))}


def select_candidates(rows):
    """Only validation RMSE enters selection; require the same three seeds."""
    groups = []
    for model, rates in GRID.items():
        for rate in rates:
            selected = [r for r in rows if r["model"] == model and r["learning_rate"] == rate]
            if sorted(r["seed"] for r in selected) != SEEDS:
                raise ValueError("Incomplete or duplicate seed set for clean-model selection.")
            values = [r["best_validation_rmse"] for r in selected]
            groups.append({"model": model, "learning_rate": rate,
                           "validation_rmse_mean": float(np.mean(values)),
                           "validation_rmse_std": float(np.std(values, ddof=1)),
                           "best_epochs": [r["best_epoch"] for r in selected],
                           "runs_at_epoch_cap": sum(r["epochs_run"] == 100 for r in selected)})
    chosen = {model: min((g for g in groups if g["model"] == model),
                        key=lambda g: (g["validation_rmse_mean"], g["learning_rate"]))
              for model in GRID}
    return groups, chosen


def tune(base, dataset, output):
    root = output / "clean_search"
    protocol = {"grid": GRID, "seeds": SEEDS, "epochs": 100, "patience": 10,
                "selection": "minimum mean best validation RMSE across fixed seeds; lower LR breaks exact ties",
                "dataset_id": dataset["manifest"]["dataset_id"],
                "test_used_for_selection": False, "base_config": base,
                "source_hashes": {n: file_hash(Path(__file__).with_name(n))
                                  for n in ("models.py", "training.py", "preflight.py")}}
    path = root / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Search protocol changed; use a new output directory.")
    write_json(path, protocol)
    rows = []
    mean = float(dataset["train"][:, 2].mean())
    manifest = dataset["manifest"]
    for model, rates in GRID.items():
        for rate in rates:
            config = copy.deepcopy(base)
            config["training"].update(epochs=100, patience=10, learning_rate=rate)
            for seed in SEEDS:
                directory = root / model / f"lr_{rate:g}" / str(seed)
                print(f"Clean search: {model}, lr={rate:g}, seed={seed}", flush=True)
                fit_model(model, dataset["train"], dataset["validation"], manifest["users"],
                          manifest["items"], config, mean, seed, directory,
                          {"dataset_id": manifest["dataset_id"], "mode": "clean"})
                record = read_json(directory / "training.json")
                rows.append({"model": model, "learning_rate": rate, "seed": seed,
                             "best_epoch": record["best_epoch"], "epochs_run": len(record["history"]),
                             "best_validation_rmse": record["best_validation_rmse_raw"],
                             "elapsed_seconds": record["elapsed_seconds"]})
                write_csv(root / "runs.csv", rows, list(rows[0]))
    groups, chosen = select_candidates(rows)
    write_json(root / "selection.json", {"candidates": groups, "selected": chosen,
               "scope": "limited learning-rate search; no test or attack metrics used"})
    configs = output / "selected_configs"
    configs.mkdir(parents=True, exist_ok=True)
    for model, selection in chosen.items():
        config = copy.deepcopy(base)
        config.pop("_config_path", None)
        config["name"] = f"validation_selected_{model}"
        config["training"].update(epochs=100, patience=10, learning_rate=selection["learning_rate"])
        config["experiment"].update(models=[model], seeds=SEEDS,
            output=str(output / f"future_experiments_{model}"))
        # Isolate future generation from completed pilot artifacts as well.
        config["agent"]["output"] = str(output / "future_agent_data")
        (configs / f"{model}.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")
    plot_training(root)
    return chosen


def plot_training(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, (model, rates) in zip(axes, GRID.items()):
        for rate in rates:
            record = read_json(root / model / f"lr_{rate:g}" / "11" / "training.json")
            ax.plot([r["epoch"] for r in record["history"]],
                    [r["validation_rmse_raw"] for r in record["history"]], label=f"lr={rate:g}")
        ax.set(title=f"{model.upper()} clean validation (seed 11)", xlabel="Epoch", ylabel="Validation RMSE")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(root / "validation_curves.png", dpi=160)
    plt.close(fig)


def load_checkpoint(directory, dataset):
    """Read-only checkpoint load with the same source/config fingerprint as training."""
    record = read_json(directory / "training.json")
    users = dataset["manifest"]["users"]
    if record["identity"]["mode"] == "poison":
        attack_id = record["identity"]["attack_id"]
        # Saved model shape is checked again against the original result metadata by diagnose().
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        users = saved["state_dict"]["user.weight"].shape[0]
    else:
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if record["identity"]["dataset_id"] != dataset["manifest"]["dataset_id"]:
        raise ValueError("Checkpoint dataset mismatch.")
    mean = float(dataset["train"][:, 2].mean())
    fingerprint = digest({"identity": record["identity"], "model": record["model"],
        "settings": record["settings"], "users": users, "items": dataset["manifest"]["items"],
        "mean": mean, "seed": record["seed"],
        "source_hashes": {n: file_hash(Path(__file__).with_name(n)) for n in ("models.py", "training.py")}})
    if fingerprint != saved["fingerprint"] or fingerprint != record["fingerprint"]:
        raise ValueError("Checkpoint fingerprint mismatch; original source/settings required.")
    seed_everything(record["seed"], record["settings"].get("threads", 4))
    model = build_model(record["model"], users, dataset["manifest"]["items"], record["settings"], mean, record["seed"])
    model.load_state_dict(saved["state_dict"])
    return model, record


def diagnose(config, dataset, output):
    pilot = resolve(config, config["experiment"]["output"]) / "experiments"
    original = read_csv(pilot / "results.csv")
    if not original or any(r["mock_validation"].lower() != "false" for r in original):
        raise ValueError("Expected completed real pilot results.")
    if any(r["dataset_id"] != dataset["manifest"]["dataset_id"] for r in original):
        raise ValueError("Pilot dataset mismatch.")
    root = output / "pilot_rankings"
    summaries = []
    for model_name, seed, target in sorted({(r["model"], int(r["seed"]), int(r["target"])) for r in original}):
        group = [r for r in original if (r["model"], int(r["seed"]), int(r["target"])) == (model_name, seed, target)]
        clean = None
        for method, row in [("clean", None)] + [(r["method"], r) for r in group]:
            directory = (pilot / "clean" / model_name / str(seed) if row is None else
                pilot / "poison" / model_name / f"target_{target}" / method / row["attack_id"][:12] / str(seed))
            model, record = load_checkpoint(directory, dataset)
            if row is not None and (record["identity"]["attack_id"] != row["attack_id"] or
                    model.user.num_embeddings != dataset["manifest"]["users"] + int(row["fake_users"])):
                raise ValueError("Pilot attack/checkpoint identity mismatch.")
            identity = {"checkpoint_sha256": file_hash(directory / "model.pt"),
                        "dataset_id": dataset["manifest"]["dataset_id"], "target": target,
                        "evaluation_settings": config["experiment"],
                        "source_sha256": file_hash(Path(__file__))}
            dest = root / model_name / str(seed) / f"target_{target}" / method
            cache = dest / "per_user.npz"
            print(f"Ranking diagnostic: {model_name}, {method}, target={target}, seed={seed}", flush=True)
            if cache.exists() and read_json(dest / "identity.json") == identity:
                with np.load(cache, allow_pickle=False) as arrays:
                    result = {k: arrays[k] for k in ("users", "ranks", "scores")}
                    result["gaps"] = {k: arrays[f"gap_{k}"] for k in (10, 50, 100)}
            else:
                evaluation_config = copy.deepcopy(config)
                evaluation_config["training"] = record["settings"]
                # Evaluate on CPU as the original pilot config does.
                result = rank_catalog(model, dataset, evaluation_config, target)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, users=result["users"], ranks=result["ranks"], scores=result["scores"],
                                    **{f"gap_{k}": v for k, v in result["gaps"].items()})
                write_json(dest / "identity.json", identity)
            summary = {"model": model_name, "method": method, "seed": seed, "target": target,
                       **summarize_ranks(result["ranks"], result["scores"], result["gaps"])}
            if method == "clean":
                clean = result
                expected_hr = float(group[0]["clean_hr"])
            else:
                summary.update(compare_ranks(clean, result))
                expected_hr = float(row["poison_hr"])
                if not np.isclose(summary["signed_prediction_shift"], float(row["signed_prediction_shift"]), atol=1e-7, rtol=1e-6):
                    raise ValueError("Recomputed shift does not reproduce pilot result.")
            if summary["hr_at_10"] != expected_hr:
                raise ValueError("Recomputed HR@10 does not reproduce pilot result.")
            user_rows = []
            for i, user in enumerate(result["users"]):
                item = {"user_id": int(user), "rank": int(result["ranks"][i]),
                        "target_score": float(result["scores"][i]),
                        **{f"score_gap_top_{k}": float(v[i]) for k, v in result["gaps"].items()}}
                if method != "clean":
                    item["rank_improvement"] = int(clean["ranks"][i] - result["ranks"][i])
                user_rows.append(item)
            write_csv(dest / "per_user.csv", user_rows, list(user_rows[0]))
            write_json(dest / "summary.json", summary)
            summaries.append(summary)
            fields = list(dict.fromkeys(k for s in summaries for k in s))
            write_csv(root / "summary.csv", summaries, fields)
    write_json(root / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deepseek_pilot.yaml")
    parser.add_argument("--output", default="runs/preflight_20261102")
    parser.add_argument("--stage", choices=("all", "tune", "diagnose"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    output = resolve(config, args.output).resolve()
    pilot = resolve(config, config["experiment"]["output"]).resolve()
    if output == pilot or pilot in output.parents or output in pilot.parents:
        raise ValueError("Preflight output must be separate from the original pilot tree.")
    dataset = load_dataset(config)
    preserved = [p for p in pilot.rglob("*") if p.is_file()]
    agent_root = resolve(config, config["agent"]["output"])
    preserved += [p for p in agent_root.rglob("*") if p.is_file()]
    before = {str(p): file_hash(p) for p in preserved}
    started = time.monotonic()
    selected = tune(config, dataset, output) if args.stage in ("all", "tune") else None
    summaries = diagnose(config, dataset, output) if args.stage in ("all", "diagnose") else None
    after = {str(p): file_hash(p) for p in preserved}
    if before != after:
        raise ValueError("Original pilot files changed during preflight.")
    write_json(output / f"completion_{args.stage}.json", {
        "stage": args.stage, "selected": selected, "diagnostic_rows": len(summaries or []),
        "elapsed_seconds": time.monotonic() - started, "original_files_unchanged": True,
        "preserved_input_hashes": before, "api_calls": 0,
        "note": "Diagnostic pilot uses original hyperparameters, not selected hyperparameters."})
    print(f"Preflight complete: {output}", flush=True)


if __name__ == "__main__":
    main()
