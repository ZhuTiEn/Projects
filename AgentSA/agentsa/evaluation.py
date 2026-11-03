"""Full-catalog target ranking; score ties broken by increasing item ID."""
from __future__ import annotations

import numpy as np

from .training import predict


def prediction_metrics(predictions, observed):
    difference = np.asarray(predictions) - np.asarray(observed)
    return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mae": float(np.mean(np.abs(difference)))}


def train_histories(train, users):
    histories = [set() for _ in range(users)]
    for u, i, _ in train:
        histories[int(u)].add(int(i))
    return histories


def target_rank_from_scores(scores, target, seen):
    """Rank 1 = best. Seen items excluded; target must be unseen."""
    if target in seen:
        raise ValueError("Target ranking requested for a seen target.")
    scores = np.asarray(scores)
    eligible = np.ones(len(scores), dtype=bool)
    eligible[list(seen)] = False
    ahead = (scores > scores[target]) | ((scores == scores[target]) & (np.arange(len(scores)) < target))
    return int(1 + np.count_nonzero(eligible & ahead))


def evaluate_target(model, target, train, test, users, items, config):
    settings = config["experiment"]
    device = config["training"]["device"]
    clip = settings["clip_predictions"]
    histories = train_histories(train, users)
    eligible = np.asarray([u for u in range(users) if target not in histories[u]], dtype=np.int64)
    if not len(eligible):
        raise ValueError("No real users eligible for target ranking.")
    ranks, scores = [], []
    catalog = np.arange(items)
    for start in range(0, len(eligible), settings["ranking_user_batch"]):
        user_batch = eligible[start:start + settings["ranking_user_batch"]]
        pairs = np.column_stack([np.repeat(user_batch, items), np.tile(catalog, len(user_batch))])
        values = predict(model, pairs, device, settings["prediction_batch"], clip).reshape(-1, items)
        for user, value in zip(user_batch, values):
            ranks.append(target_rank_from_scores(value, target, histories[user]))
            scores.append(float(value[target]))
    test_prediction = predict(model, test, device, settings["prediction_batch"], clip)
    metrics = prediction_metrics(test_prediction, test[:, 2])
    metrics.update(hr=float(np.mean(np.asarray(ranks) <= settings["top_k"])),
                   eligible_users=len(eligible), mean_target_score=float(np.mean(scores)))
    for k in (10, 50, 100):
        metrics[f"hr_at_{k}"] = float(np.mean(np.asarray(ranks) <= k))
    metrics.update(mean_target_rank=float(np.mean(ranks)), median_target_rank=float(np.median(ranks)))
    return metrics, np.asarray(scores), np.asarray(ranks)
