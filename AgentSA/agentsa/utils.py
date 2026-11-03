from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent:
        config = merge_dict(load_config(path.parent / parent), config)
    root = Path(config.get("project_root", ".."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    config["project_root"] = str(root)
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config):
    data = config["data"]
    if not (0 < data["train_fraction"] < 1 and 0 < data["validation_fraction"] < 1
            and data["train_fraction"] + data["validation_fraction"] < 1):
        raise ValueError("Train/validation fractions must leave a nonempty test fraction.")
    for value in (data["min_interactions"], data["target_count"], config["training"]["epochs"],
                  config["training"]["batch_size"], config["training"]["patience"],
                  config["experiment"]["prediction_batch"], config["experiment"]["ranking_user_batch"],
                  config["experiment"]["top_k"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Counts, epochs, batch sizes and top-k must be positive integers.")
    attack, agent = config["attack"], config["agent"]
    if not 0 < attack["fake_ratio"] <= 1:
        raise ValueError("fake_ratio must be in (0,1].")
    if not 0 < attack["aop_fraction"] <= 1 or not 0 <= attack["selected_fraction"] <= 1:
        raise ValueError("Invalid AoP or selected-item fraction.")
    for value in (attack["filler_count"], agent["rounds"], agent["exposure_size"],
                  agent["max_selected_per_round"], agent["filler_cap"], agent["reference_per_round"],
                  agent["relevant_k"], agent["recent_m"]):
        if type(value) is not int or value < 0:
            raise ValueError("Agent/attack counts must be nonnegative integers.")
    if agent["profile_candidate_multiplier"] < 1 or agent["validation_retries"] < 1:
        raise ValueError("Need at least one profile candidate and validation attempt.")
    if agent["popularity_alpha"] < 0:
        raise ValueError("popularity_alpha must be nonnegative.")
    minimum = agent.get("min_selected_per_round", 0)
    if type(minimum) is not int or not 0 <= minimum <= agent["max_selected_per_round"]:
        raise ValueError("min_selected_per_round must be between zero and max_selected_per_round.")
    if agent.get("exposure_policy", "popularity") not in {"popularity", "target_aware"}:
        raise ValueError("Unknown exposure_policy.")
    if type(agent.get("semantic_pool_size", 200)) is not int or agent.get("semantic_pool_size", 200) < 1:
        raise ValueError("semantic_pool_size must be a positive integer.")
    if not 0 <= agent.get("semantic_fraction", 0.5) <= 1:
        raise ValueError("semantic_fraction must be in [0,1].")
    if agent.get("profile_diversity_mode", "legacy") not in {"legacy", "strict_v1"}:
        raise ValueError("Unknown profile_diversity_mode.")
    if not 0 < config["experiment"].get("target_success_hr", 0.01) <= 1:
        raise ValueError("target_success_hr must be in (0,1].")
    if "reviews" in config:
        reviews = config["reviews"]
        for key in ("data_output", "agent_output", "experiment_output", "encoder", "training"):
            if key not in reviews:
                raise ValueError(f"reviews.{key} is required.")
        if reviews["encoder"] not in {"hash", "sentence_transformer"}:
            raise ValueError("reviews.encoder must be hash or sentence_transformer.")
        if reviews.get("protocol", "legacy") not in {"legacy", "strict_v2", "strict_v3"}:
            raise ValueError("Unknown reviews.protocol.")
        if type(reviews["validation_retries"]) is not int or reviews["validation_retries"] < 1:
            raise ValueError("reviews.validation_retries must be positive.")
        if type(reviews["recent_reviews"]) is not int or reviews["recent_reviews"] < 0:
            raise ValueError("reviews.recent_reviews must be nonnegative.")
        training = reviews["training"]
        for key in ("epochs", "patience", "batch_size", "id_dimensions", "hidden_dimensions"):
            if type(training[key]) is not int or training[key] <= 0:
                raise ValueError(f"reviews.training.{key} must be a positive integer.")
        if training["learning_rate"] <= 0 or training["l2"] < 0:
            raise ValueError("Invalid review-model learning rate or L2.")


def resolve(config, path):
    path = Path(path)
    return path if path.is_absolute() else Path(config["project_root"]) / path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rating_array(path):
    rows = read_csv(path)
    return np.asarray([[int(r["user_id"]), int(r["item_id"]), float(r["rating"])]
                       for r in rows], dtype=np.float64).reshape(-1, 3)


def stable_seed(seed, *parts):
    return int(digest([seed, *parts])[:8], 16)


def seed_everything(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
