from __future__ import annotations

import math

import numpy as np

from .utils import read_csv, write_csv

METHODS = {"target_only", "random", "average", "bandwagon", "segmented", "mixed", "aop"}


def fake_user_count(users, settings):
    count = math.floor(users * settings["fake_ratio"])
    if count < 1:
        raise ValueError("Fake-user ratio gives zero fake users; increase data size or ratio.")
    return count


def statistics(train, items):
    ids = train[:, 1].astype(int)
    counts = np.bincount(ids, minlength=items)
    sums = np.bincount(ids, weights=train[:, 2], minlength=items)
    squares = np.bincount(ids, weights=train[:, 2] ** 2, minlength=items)
    means = np.divide(sums, counts, out=np.full(items, train[:, 2].mean()), where=counts > 0)
    variance = np.divide(squares, counts, out=means ** 2, where=counts > 0) - means ** 2
    return counts, means, np.sqrt(np.maximum(variance, 0))


def traditional_attack(method, train, real_users, items, target, count, lengths, settings, seed):
    if method not in METHODS:
        raise ValueError(f"Unknown attack: {method}")
    rng = np.random.default_rng(seed)
    counts, means, stds = statistics(train, items)
    popularity = np.lexsort((np.arange(items), -counts))
    available = np.asarray([i for i in popularity if i != target], dtype=int)
    popular = np.asarray([i for i in popularity[:max(1, math.ceil(items * settings["aop_fraction"]))]
                          if i != target], dtype=int)
    rows = []
    mixture = ["random", "average", "bandwagon", "segmented"]
    for a in range(count):
        user = real_users + a
        rows.append({"user_id": user, "item_id": target, "rating": 5})
        if method == "target_only":
            continue
        current = mixture[a % 4] if method == "mixed" else method
        length = int(lengths[a])
        pool = popular if current == "aop" else available
        if length > len(pool) or length < 0:
            raise ValueError(f"Requested {length} filler items, but {current} pool has {len(pool)}.")
        selected_count = 0
        selected = np.empty(0, dtype=int)
        if current in {"bandwagon", "segmented"} and length:
            selected_count = min(length, max(1, math.floor(length * settings["selected_fraction"])))
            selected = available[:selected_count]
        remaining = np.asarray([i for i in pool if i not in set(selected)], dtype=int)
        fillers = rng.choice(remaining, length - selected_count, replace=False)
        for item in selected:
            rows.append({"user_id": user, "item_id": int(item), "rating": 5})
        for item in fillers:
            if current == "segmented":
                rating = 1
            elif current in {"average", "aop"}:
                rating = int(np.clip(np.rint(rng.normal(means[item], stds[item])), 1, 5))
            else:
                rating = int(np.clip(np.rint(rng.normal(train[:, 2].mean(), train[:, 2].std())), 1, 5))
            rows.append({"user_id": user, "item_id": int(item), "rating": rating})
    validate_fake(rows, real_users, items, target, count)
    return rows


def validate_fake(rows, real_users, items, target, count):
    pairs, target_users = set(), set()
    expected = set(range(real_users, real_users + count))
    for row in rows:
        user, item, rating = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
        if user not in expected or not 0 <= item < items or not 1 <= rating <= 5 or not np.isfinite(rating):
            raise ValueError("Invalid fake user/item/rating.")
        if (user, item) in pairs:
            raise ValueError("Repeated fake user-item interaction.")
        pairs.add((user, item))
        if item == target:
            if rating != 5:
                raise ValueError("Push target rating must be 5.")
            target_users.add(user)
    if target_users != expected:
        raise ValueError("Every fake user must rate the target once.")


def save_attack(path, rows):
    write_csv(path, rows, ["user_id", "item_id", "rating"])


def load_attack(path):
    return [{"user_id": int(r["user_id"]), "item_id": int(r["item_id"]), "rating": float(r["rating"])}
            for r in read_csv(path)]


def as_array(rows):
    return np.asarray([[r["user_id"], r["item_id"], r["rating"]] for r in rows], dtype=float).reshape(-1, 3)
