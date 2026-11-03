"""Independent review attack generation; never changes rating-Agent identity."""
from __future__ import annotations

import json
from pathlib import Path

from .agents import agent_directory, public_item
from .attacks import load_attack
from .llm import ChatClient
from .utils import (append_jsonl, canonical, digest, file_hash, read_json,
                    resolve, write_json)


REVIEW_PROMPTS = {
    "review_analyze": ("Extract only target features supported by the supplied metadata. Return "
                "{features:[{feature:string,evidence:string}],use_cases:[strings]}."),
    "target_review": ("Write one natural positive review consistent with the profile and rating. Return "
               "{review:string}. Do not mention experiments or recommendation systems."),
    "plain_reviews": ("Write one natural review for every supplied item, consistent with its rating, profile and memory. "
              "Return {reviews:[{item_id:integer,review:string}]}. Use only supplied item IDs."),
    "propagate_reviews": ("For each item, identify only genuinely shared features supported by both target and filler "
                  "metadata, then subtly revise the initial review. Return "
                  "{reviews:[{item_id:integer,shared_features:[{feature:string,target_evidence:string,"
                  "filler_evidence:string}],review:string}]}. If no supported overlap exists, use an empty "
                  "shared_features list and copy the initial review. Preserve rating sentiment; never name the "
                  "target item, attack, experiment, or recommendation system."),
}


class MockReviewClient:
    identity = {"provider": "mock-review", "model": "deterministic-fixture"}

    def generate(self, task, instruction, payload):
        if task == "review_analyze":
            title = payload["target"].get("title") or "product"
            return {"features": [{"feature": "practical design", "evidence": title}],
                    "use_cases": ["routine pet care"]}
        if task == "target_review":
            return {"review": "A practical product that worked very well for my routine."}
        if task == "plain_reviews":
            return {"reviews": [{"item_id": row["item_id"],
                                  "review": f"A solid item that matches my {row['rating']}-star rating."}
                                 for row in payload["items"]]}
        if task == "propagate_reviews":
            return {"reviews": [{"item_id": row["item_id"],
                                  "shared_features": [{"feature": "practical design",
                                    "target_evidence": "target metadata",
                                    "filler_evidence": "filler metadata"}],
                                  "review": row["initial_review"] + " Its practical design is useful."}
                                 for row in payload["items"]]}
        raise ValueError(f"Unknown mock review task: {task}")


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def valid_analysis(value):
    features = value["features"]
    if not isinstance(features, list) or not features:
        raise ValueError("Target analysis needs supported features.")
    cleaned = []
    for row in features:
        cleaned.append({"feature": _nonempty(row["feature"], "feature"),
                        "evidence": _nonempty(row["evidence"], "evidence")})
    uses = value.get("use_cases", [])
    if not isinstance(uses, list) or any(not isinstance(x, str) for x in uses):
        raise ValueError("use_cases must be strings.")
    return {"features": cleaned, "use_cases": [x.strip() for x in uses if x.strip()]}


def valid_review_batch(value, allowed, field="reviews"):
    rows = value[field]
    if not isinstance(rows, list) or len(rows) != len(allowed):
        raise ValueError("Review batch must cover every supplied item exactly once.")
    result, ids = [], []
    for row in rows:
        if type(row.get("item_id")) is not int:
            raise ValueError("Review item IDs must be integers.")
        ids.append(row["item_id"])
        result.append({"item_id": row["item_id"],
                       "review": _nonempty(row.get("review"), "review")})
    if len(set(ids)) != len(ids) or set(ids) != set(allowed):
        raise ValueError("Review IDs differ from the supplied items.")
    return result


def valid_propagated(value, allowed, target):
    rows = valid_review_batch(value, allowed)
    source = {row["item_id"]: row for row in value["reviews"]}
    forbidden = [str(target.get("title") or "").strip(), str(target.get("asin") or "").strip()]
    result = []
    for row in rows:
        raw_features = source[row["item_id"]].get("shared_features", [])
        if not isinstance(raw_features, list):
            raise ValueError("shared_features must be a list.")
        features = []
        for feature in raw_features:
            features.append({key: _nonempty(feature.get(key), key) for key in
                             ("feature", "target_evidence", "filler_evidence")})
        lowered = row["review"].casefold()
        if any(value and value.casefold() in lowered for value in forbidden):
            raise ValueError("Propagated review directly names the target item.")
        result.append({**row, "shared_features": features})
    return result


def _actions(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def review_directory(config, target, mock=False):
    return resolve(config, config["reviews"]["agent_output"]) / ("mock" if mock else "real") / f"target_{target}"


def _request(client, task, payload, validator, retries):
    last = None
    for attempt in range(retries):
        try:
            value = client.generate(task, REVIEW_PROMPTS[task],
                                    {**payload, "schema_attempt": attempt})
            return validator(value)
        except (ValueError, KeyError, TypeError) as exc:
            last = exc
    raise ValueError(f"Invalid {task} response after {retries} attempts: {last}")


def generate_review_attack(config, dataset, target, mock=False):
    source = agent_directory(config, target, mock)
    source_manifest = read_json(source / "manifest.json")
    if source_manifest["dataset_id"] != dataset["manifest"]["dataset_id"]:
        raise ValueError("Rating Agent belongs to another dataset.")
    profiles = read_json(source / "profiles.json")
    ratings = load_attack(source / "fake_ratings.csv")
    action_rows = _actions(source / "actions.jsonl")
    output = review_directory(config, target, mock)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".generation.lock"
    try:
        lock_handle = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError("Review generation is already running or left a lock.") from exc
    try:
        return _generate(config, dataset, target, mock, output, source, source_manifest,
                         profiles, ratings, action_rows)
    finally:
        lock_handle.close()
        lock.unlink(missing_ok=True)


def _generate(config, dataset, target, mock, output, source, source_manifest,
              profiles, ratings, action_rows):
    client = MockReviewClient() if mock else ChatClient(config["llm"], output / "llm")
    settings = config["reviews"]
    signature = digest({"dataset_id": dataset["manifest"]["dataset_id"],
                        "rating_manifest_sha256": file_hash(source / "manifest.json"),
                        "settings": settings, "client": client.identity,
                        "prompts": REVIEW_PROMPTS})
    identity = output / "identity.json"
    if identity.exists() and read_json(identity)["signature"] != signature:
        raise ValueError("Review settings/model changed; choose a new reviews.agent_output directory.")
    write_json(identity, {"signature": signature, "settings": settings,
                          "client": client.identity, "prompts": REVIEW_PROMPTS})
    target_item = public_item(dataset["items"][target])
    analysis_path = output / "target_analysis.json"
    if analysis_path.exists():
        analysis = read_json(analysis_path)
    else:
        analysis = _request(client, "review_analyze", {"target": target_item}, valid_analysis,
                            settings["validation_retries"])
        write_json(analysis_path, analysis)
    by_user = {}
    for row in ratings:
        by_user.setdefault(row["user_id"], {})[row["item_id"]] = row["rating"]
    actions = {}
    for row in action_rows:
        actions[(int(row["user_id"]), int(row["round"]))] = row
    real_users = dataset["manifest"]["users"]
    for index, profile in enumerate(profiles):
        user = real_users + index
        state_path = output / "states" / f"{user}.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["signature"] != signature or state["profile_hash"] != digest(profile):
                raise ValueError("Review checkpoint identity/profile changed.")
        else:
            target_review = _request(client, "target_review",
                {"profile": profile, "target": target_item, "rating": 5},
                lambda value: _nonempty(value["review"], "target review"),
                settings["validation_retries"])
            state = {"signature": signature, "profile_hash": digest(profile), "next_round": 0,
                     "plain": [{"user_id": user, "item_id": target, "rating": 5,
                                "round": -1, "review": target_review}],
                     "tfp": [{"user_id": user, "item_id": target, "rating": 5,
                              "round": -1, "review": target_review, "shared_features": []}]}
            write_json(state_path, state)
        for round_id in range(state["next_round"], config["agent"]["rounds"]):
            action = actions.get((user, round_id))
            if action is None:
                raise ValueError(f"Missing rating action for user {user}, round {round_id}.")
            selected = [int(row["item_id"]) for row in action["ratings"]]
            supplied = [{**public_item(dataset["items"][item]),
                         "rating": int(by_user[user][item])} for item in selected]
            memory = [{"item_id": row["item_id"], "rating": row["rating"],
                       "review": row["review"]} for row in state["plain"][-settings["recent_reviews"]:]]
            plain = _request(client, "plain_reviews",
                {"profile": profile, "memory": memory, "items": supplied},
                lambda value, selected=selected: valid_review_batch(value, selected),
                settings["validation_retries"])
            initial_by_item = {row["item_id"]: row["review"] for row in plain}
            propagation_items = [{**item, "initial_review": initial_by_item[item["item_id"]]}
                                 for item in supplied]
            propagated = _request(client, "propagate_reviews",
                {"profile": profile, "target": target_item, "target_analysis": analysis,
                 "items": propagation_items},
                lambda value, selected=selected: valid_propagated(value, selected, target_item),
                settings["validation_retries"])
            for row in plain:
                state["plain"].append({"user_id": user, "item_id": row["item_id"],
                                       "rating": by_user[user][row["item_id"]], "round": round_id,
                                       "review": row["review"]})
            for row in propagated:
                state["tfp"].append({"user_id": user, "item_id": row["item_id"],
                                     "rating": by_user[user][row["item_id"]], "round": round_id,
                                     "initial_review": initial_by_item[row["item_id"]],
                                     "review": row["review"],
                                     "shared_features": row["shared_features"]})
            state["next_round"] = round_id + 1
            write_json(state_path, state)
    states = [read_json(output / "states" / f"{real_users + i}.json")
              for i in range(len(profiles))]
    plain_rows = [row for state in states for row in state["plain"]]
    tfp_rows = [row for state in states for row in state["tfp"]]
    for name, rows in (("plain_reviews.jsonl", plain_rows), ("tfp_reviews.jsonl", tfp_rows)):
        path = output / name
        path.unlink(missing_ok=True)
        for row in rows:
            append_jsonl(path, row)
    manifest = {"signature": signature, "dataset_id": dataset["manifest"]["dataset_id"],
                "target": target, "mock": mock, "fake_users": len(profiles),
                "rating_manifest_sha256": file_hash(source / "manifest.json"),
                "rating_csv_sha256": source_manifest["sha256"],
                "plain_interactions": len(plain_rows), "tfp_interactions": len(tfp_rows),
                "plain_sha256": file_hash(output / "plain_reviews.jsonl"),
                "tfp_sha256": file_hash(output / "tfp_reviews.jsonl"),
                "causal_design": "identical users/items/ratings; TFP changes filler review text only",
                "scope": "post-hoc review generation over a fixed rating-Agent trajectory"}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def load_generated_reviews(config, dataset, target, mode, mock=False):
    if mode not in {"plain", "tfp"}:
        raise ValueError("Review mode must be plain or tfp.")
    directory = review_directory(config, target, mock)
    manifest = read_json(directory / "manifest.json")
    if manifest["dataset_id"] != dataset["manifest"]["dataset_id"] or manifest["target"] != target:
        raise ValueError("Generated reviews belong to another dataset/target.")
    rating_dir = agent_directory(config, target, mock)
    if (file_hash(rating_dir / "manifest.json") != manifest["rating_manifest_sha256"] or
            file_hash(rating_dir / "fake_ratings.csv") != manifest["rating_csv_sha256"]):
        raise ValueError("The source rating Agent changed after review generation.")
    path = directory / f"{mode}_reviews.jsonl"
    if file_hash(path) != manifest[f"{mode}_sha256"]:
        raise ValueError("Generated review file hash mismatch.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()], manifest
