"""Review-text sidecar aligned exactly with the frozen rating splits."""
from __future__ import annotations

from pathlib import Path

from .data import records
from .utils import (append_jsonl, digest, file_hash, read_csv, read_json,
                    resolve, write_json)


def _text(value):
    return str(value or "").strip()


def _split_pairs(directory, name):
    return {(int(row["user_id"]), int(row["item_id"])): row
            for row in read_csv(directory / f"{name}.csv")}


def _data_settings(settings):
    """Only fields that affect sidecar bytes; model/output changes may reuse it."""
    return {"data_output": settings["data_output"], "schema_version": 1}


def prepare_review_data(config, dataset):
    """Save review text without changing rating IDs, splits, targets, or raw files."""
    settings = config["reviews"]
    output = resolve(config, settings["data_output"])
    rating_root = resolve(config, config["data"]["output"])
    if (output / "manifest.json").exists():
        return load_review_data(config, dataset)["manifest"]

    id_maps = read_json(rating_root / "id_maps.json")
    users, items = id_maps["users"], id_maps["items"]
    splits = {name: _split_pairs(rating_root, name)
              for name in ("train", "validation", "test")}
    pair_to_split = {}
    for name, pairs in splits.items():
        for pair in pairs:
            if pair in pair_to_split:
                raise ValueError("Rating pair occurs in multiple splits.")
            pair_to_split[pair] = name

    latest = {}
    for raw in records(resolve(config, config["data"]["reviews"])):
        raw_user, raw_item = str(raw["reviewerID"]), str(raw["asin"])
        if raw_user not in users or raw_item not in items:
            continue
        pair = (int(users[raw_user]), int(items[raw_item]))
        if pair not in pair_to_split:
            continue
        timestamp = int(raw["unixReviewTime"])
        row = {"user_id": pair[0], "item_id": pair[1],
               "rating": float(raw["overall"]), "timestamp": timestamp,
               "summary": _text(raw.get("summary")),
               "review": _text(raw.get("reviewText"))}
        if pair not in latest or timestamp >= latest[pair]["timestamp"]:
            latest[pair] = row

    if set(latest) != set(pair_to_split):
        missing = len(set(pair_to_split) - set(latest))
        raise ValueError(f"Review sidecar could not recover {missing} rating pairs.")
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train", "validation", "test"):
        path = output / f"{name}_reviews.jsonl"
        path.unlink(missing_ok=True)
        rows = [latest[pair] for pair in sorted(splits[name])]
        for row in rows:
            saved = splits[name][(row["user_id"], row["item_id"])]
            if float(saved["rating"]) != row["rating"] or int(saved["timestamp"]) != row["timestamp"]:
                raise ValueError("Review sidecar rating/timestamp differs from the frozen split.")
        for row in rows:
            stored = {**row, "split": name}
            append_jsonl(path, stored)
        counts[name] = len(rows)
    files = {f"{name}_reviews.jsonl": file_hash(output / f"{name}_reviews.jsonl")
             for name in ("train", "validation", "test")}
    manifest = {
        "dataset_id": dataset["manifest"]["dataset_id"],
        "settings": _data_settings(settings),
        "counts": counts,
        "empty_reviews": sum(not row["review"] for row in latest.values()),
        "raw_reviews_sha256": dataset["manifest"]["raw_files"]["reviews"]["sha256"],
        "files": files,
        "identity": digest({"dataset_id": dataset["manifest"]["dataset_id"],
                            "settings": _data_settings(settings), "files": files}),
        "scope": "text sidecar only; exact frozen rating pairs and splits",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _read_jsonl(path):
    import json
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_review_data(config, dataset):
    output = resolve(config, config["reviews"]["data_output"])
    manifest = read_json(output / "manifest.json")
    if manifest["dataset_id"] != dataset["manifest"]["dataset_id"]:
        raise ValueError("Review data belongs to another rating dataset.")
    # Older manifests stored the entire reviews section; only data_output ever
    # affected sidecar bytes. Accept them while enforcing the actual path.
    if manifest["settings"].get("data_output") != config["reviews"]["data_output"]:
        raise ValueError("Review data output differs from the current configuration.")
    for name, expected in manifest["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Prepared review file changed: {name}")
    return {"manifest": manifest,
            **{name: _read_jsonl(output / f"{name}_reviews.jsonl")
               for name in ("train", "validation", "test")}}
