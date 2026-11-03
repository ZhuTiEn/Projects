"""Streaming Amazon readers; raw archives are always read-only."""
from __future__ import annotations

import ast
import gzip
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .utils import (digest, file_hash, rating_array, read_json, resolve,
                    write_csv, write_json)


def records(path):
    import json
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)  # Historical metadata uses Python literals.
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                yield row
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid record: {path}, line {number}") from exc


def k_core(rows, minimum):
    while True:
        uc = Counter(r["raw_user"] for r in rows)
        ic = Counter(r["raw_item"] for r in rows)
        kept = [r for r in rows if uc[r["raw_user"]] >= minimum and ic[r["raw_item"]] >= minimum]
        if len(kept) == len(rows):
            return kept
        rows = kept


def split_rows(rows, rng, train_fraction, validation_fraction):
    """Global random split, followed by a logged warm-start coverage repair."""
    indices = rng.permutation(len(rows))
    a = int(len(rows) * train_fraction)
    b = int(len(rows) * (train_fraction + validation_fraction))
    train = [rows[i] for i in indices[:a]]
    validation = [rows[i] for i in indices[a:b]]
    test = [rows[i] for i in indices[b:]]
    users = {r["user_id"] for r in train}
    items = {r["item_id"] for r in train}
    moved = {"validation": 0, "test": 0}
    for name, part in [("validation", validation), ("test", test)]:
        keep = []
        for row in part:
            if row["user_id"] not in users or row["item_id"] not in items:
                train.append(row)
                users.add(row["user_id"])
                items.add(row["item_id"])
                moved[name] += 1
            else:
                keep.append(row)
        part[:] = keep
    if not train or not validation or not test:
        raise ValueError("Empty split after coverage repair; use more data.")
    return train, validation, test, moved


def flatten_text(value):
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_text(x) for x in value)
    return str(value or "").strip()


def prepare(config):
    settings = config["data"]
    out = resolve(config, settings["output"])
    if (out / "manifest.json").exists():
        previous = read_json(out / "manifest.json")
        if previous["settings"] != settings:
            raise ValueError("Prepared data settings differ. Choose a new data.output directory.")
        for label in ("reviews", "metadata"):
            if file_hash(resolve(config, settings[label])) != previous["raw_files"][label]["sha256"]:
                raise ValueError("Raw archive changed. Use a new output directory.")
        load_dataset(config)  # Verify saved splits have not changed.
        print(f"Verified existing prepared data: {out}")
        return previous
    rng = np.random.default_rng(settings["seed"])
    deduplicated = {}
    years = Counter()
    input_rows = 0
    for r in records(resolve(config, settings["reviews"])):
        input_rows += 1
        timestamp = int(r["unixReviewTime"])
        year = datetime.fromtimestamp(timestamp, timezone.utc).year
        years[year] += 1
        if settings.get("year") is not None and year != int(settings["year"]):
            continue
        rating = float(r["overall"])
        if not 1 <= rating <= 5:
            raise ValueError("Rating outside [1,5].")
        key = (str(r["reviewerID"]), str(r["asin"]))
        row = dict(raw_user=key[0], raw_item=key[1], rating=rating, timestamp=timestamp)
        if key not in deduplicated or timestamp >= deduplicated[key]["timestamp"]:
            deduplicated[key] = row
    rows = list(deduplicated.values())
    if settings.get("max_users"):
        users = sorted({r["raw_user"] for r in rows})
        chosen = set(rng.choice(users, min(len(users), settings["max_users"]), replace=False))
        rows = [r for r in rows if r["raw_user"] in chosen]
    rows = k_core(rows, settings["min_interactions"])
    if len(rows) < 20:
        raise ValueError("Too few interactions after filtering.")
    users = {v: k for k, v in enumerate(sorted({r["raw_user"] for r in rows}))}
    item_ids = {v: k for k, v in enumerate(sorted({r["raw_item"] for r in rows}))}
    indexed = [dict(user_id=users[r["raw_user"]], item_id=item_ids[r["raw_item"]],
                    rating=r["rating"], timestamp=r["timestamp"]) for r in rows]
    indexed.sort(key=lambda r: (r["user_id"], r["item_id"]))
    train, validation, test, moved = split_rows(indexed, rng, settings["train_fraction"],
                                               settings["validation_fraction"])
    items = {idx: {"item_id": idx, "asin": asin, "title": "", "category": "", "description": ""}
             for asin, idx in item_ids.items()}
    for raw in records(resolve(config, settings["metadata"])):
        if raw.get("asin") not in item_ids:
            continue
        idx = item_ids[raw["asin"]]
        items[idx].update(title=flatten_text(raw.get("title")),
                          category=flatten_text(raw.get("categories"))[:1200],
                          description=flatten_text(raw.get("description"))[:3000])
    train_items = {r["item_id"] for r in train}
    eligible = sorted(i for i in train_items if items[i]["title"])
    if len(eligible) < settings["target_count"]:
        raise ValueError("Not enough target items with titles.")
    targets = sorted(int(i) for i in rng.choice(eligible, settings["target_count"], replace=False))
    out.mkdir(parents=True, exist_ok=True)
    fields = ["user_id", "item_id", "rating", "timestamp"]
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        write_csv(out / f"{name}.csv", part, fields)
    write_json(out / "id_maps.json", {"users": users, "items": item_ids})
    write_json(out / "items.json", [items[i] for i in range(len(items))])
    write_json(out / "targets.json", targets)
    manifest = {
        "settings": settings, "input_reviews": input_rows,
        "deduplicated_window_reviews": len(deduplicated),
        "year_counts": dict(sorted(years.items())),
        "users": len(users), "items": len(items), "interactions": len(indexed),
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "coverage_repair_moved_to_train": moved,
        "items_missing_title": sum(not v["title"] for v in items.values()),
        "targets": targets,
        "protocol": "full-window iterative k-core; global 8:1:1 then warm-start repair; UTC years",
        "raw_files": {key: {"path": str(resolve(config, settings[key])),
                            "sha256": file_hash(resolve(config, settings[key]))}
                      for key in ("reviews", "metadata")},
        "files": {name: file_hash(out / name) for name in
                  ["train.csv", "validation.csv", "test.csv", "items.json", "targets.json", "id_maps.json"]},
    }
    manifest["dataset_id"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(users)} users / {len(items)} items / {len(indexed)} interactions -> {out}")
    return manifest


def load_dataset(config):
    out = resolve(config, config["data"]["output"])
    manifest = read_json(out / "manifest.json")
    if manifest["settings"] != config["data"]:
        raise ValueError("Config does not match the prepared data manifest.")
    for name, expected in manifest["files"].items():
        if file_hash(out / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    return dict(manifest=manifest, train=rating_array(out / "train.csv"),
                validation=rating_array(out / "validation.csv"), test=rating_array(out / "test.csv"),
                items=read_json(out / "items.json"), targets=read_json(out / "targets.json"))
