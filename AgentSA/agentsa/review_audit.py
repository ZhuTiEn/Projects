"""Offline audit of generated review attacks; never contacts an LLM service."""
from __future__ import annotations

from .agents import agent_directory, profile_diversity_report, public_item
from .review_generation import (load_generated_reviews, review_directory,
                                valid_propagated)
from .utils import read_json


def _usage(path):
    if not path.exists():
        return {"http_attempts": 0, "successful_responses": 0, "total_tokens": 0}
    value = read_json(path)
    return {key: value[key] for key in ("http_attempts", "successful_responses", "total_tokens")}


def audit_review_attack(config, dataset, target, mock=False):
    rating_dir = agent_directory(config, target, mock)
    text_dir = review_directory(config, target, mock)
    profiles = read_json(rating_dir / "profiles.json")
    plain, manifest = load_generated_reviews(config, dataset, target, "plain", mock)
    tfp, _ = load_generated_reviews(config, dataset, target, "tfp", mock)
    plain_pairs = {(r["user_id"], r["item_id"], float(r["rating"])) for r in plain}
    tfp_pairs = {(r["user_id"], r["item_id"], float(r["rating"])) for r in tfp}
    target_item = {**public_item(dataset["items"][target]),
                   "asin": dataset["items"][target].get("asin", "")}
    filler = [row for row in tfp if row.get("round", -1) >= 0]
    invalid = []
    for row in filler:
        supplied = {**public_item(dataset["items"][row["item_id"]]),
                    "asin": dataset["items"][row["item_id"]].get("asin", ""),
                    "initial_review": row["initial_review"]}
        try:
            valid_propagated({"reviews": [{"item_id": row["item_id"],
                "review": row["review"], "shared_features": row["shared_features"]}]},
                [row["item_id"]], target_item, [supplied], strict=True)
        except (ValueError, KeyError, TypeError) as exc:
            invalid.append({"user_id": row["user_id"], "item_id": row["item_id"],
                            "error": str(exc)})
    profile_report = profile_diversity_report(profiles)
    shared = [row for row in filler if row.get("shared_features")]
    changed = [row for row in filler if row["review"] != row["initial_review"]]
    rating_usage = _usage(rating_dir / "llm" / "usage.json")
    review_usage = _usage(text_dir / "llm" / "usage.json")
    output = {
        "target": target, "mock": mock, "review_protocol": manifest.get("review_protocol", "legacy"),
        "fake_users": len(profiles), "profile_diversity": profile_report,
        "plain_interactions": len(plain), "tfp_interactions": len(tfp),
        "filler_reviews": len(filler), "shared_feature_reviews": len(shared),
        "changed_reviews": len(changed), "strict_validation_failures": invalid,
        "propagation_exercised": bool(shared and changed),
        "identical_rating_pairs": plain_pairs == tfp_pairs,
        "generation_validation_errors": manifest.get("validation_errors", 0),
        "rating_usage": rating_usage, "review_usage": review_usage,
    }
    output["passed"] = bool(profile_report["passed"] and not invalid and
                            plain_pairs == tfp_pairs and len(plain) == len(tfp) and
                            manifest.get("review_protocol") == "strict_v2" and
                            (mock or output["propagation_exercised"]))
    return output
