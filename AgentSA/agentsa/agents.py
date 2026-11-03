"""Rating-stage AgentSA: persistent profiles, hybrid memory, item choice and ratings."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .attacks import fake_user_count, save_attack, statistics, validate_fake
from .llm import ChatClient, MockClient
from .utils import (append_jsonl, digest, file_hash, read_json, resolve,
                    stable_seed, write_json)

PROMPTS = {
    "analyze_target": "Analyze the target product using its public metadata. Return {features:[strings], likely_preferences:[strings]}. Do not invent unsupported product facts.",
    "profile": "Infer one diverse synthetic person who would strongly prefer this target. Return {profile:{age:integer, occupation:string, interests:[strings], rating_habits:string}}. Candidate ID is for diversity, not a product ID.",
    "validate_profile": "Would a person with this profile genuinely prefer the target product? Return {prefers_target:boolean}. Evaluate consistency, not just the stated experiment goal.",
    "diversity": "Select exactly count candidate IDs with diverse demographics, interests and interaction habits while retaining target preference. Return {selected_ids:[integers]}. Select only supplied IDs, with no duplicates.",
    "select": "Act according to the supplied profile. Choose products you would evaluate from the supplied exposure list. The reference_count is a soft reference, not a mandatory count. You may choose none, at most max_count. Return {item_ids:[integers]} with unique supplied IDs.",
    "rate": "Act according to the supplied profile and your own retrieved interaction history. Rate every supplied item exactly once on the integer scale 1 to 5, based on personal preference, not blanket maximum ratings. Return {ratings:[{item_id:integer,rating:integer}]}. Do not add other items.",
}

STRICT_PROFILE_PROMPT = (
    "Infer one synthetic person who strongly prefers the target while being meaningfully different from "
    "the supplied existing_profiles. Vary age band, occupation, interests and rating habits; do not copy a "
    "previous demographic template. Return {profile:{age:integer, occupation:string, interests:[strings], "
    "rating_habits:string}}. Candidate ID is for diversity, not a product ID."
)


def active_prompt_identity(settings):
    if settings.get("profile_diversity_mode", "legacy") == "strict_v1":
        return {**PROMPTS, "profile_strict_instruction": STRICT_PROFILE_PROMPT}
    return PROMPTS


def public_item(item):
    return {"item_id": item["item_id"], "title": item["title"][:500],
            "category": item["category"][:500], "description": item["description"][:1000]}


class ItemEncoder:
    def __init__(self, items, settings, directory, mock=False):
        self.mode = settings["encoder"]
        if mock:
            self.mode = "lexical_test"
        if self.mode == "lexical_test" and not mock:
            raise ValueError("Lexical encoder is for --mock verification only; use semantic retrieval for real runs.")
        self.identity = {"mode": self.mode, "model": settings["encoder_model"],
                         "revision": settings.get("encoder_revision")}
        texts = [" ".join([x["title"], x["category"], x["description"][:1000]]) for x in items]
        signature = digest({"texts": texts, "encoder": self.identity})
        path = Path(directory) / f"item_vectors_{signature[:16]}.npy"
        if path.exists():
            self.vectors = np.load(path, allow_pickle=False)
            return
        if self.mode == "lexical_test":
            vectors = np.zeros((len(items), 256), dtype=np.float32)
            for index, text in enumerate(texts):
                for word in re.findall(r"[a-z0-9]+", text.lower()):
                    vectors[index, int(digest(word)[:8], 16) % 256] += 1
        elif self.mode == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install semantic dependencies: python -m pip install -e \".[semantic]\"") from exc
            encoder = SentenceTransformer(settings["encoder_model"],
                                          revision=settings.get("encoder_revision"),
                                          local_files_only=settings.get("encoder_local_only", False))
            vectors = encoder.encode(texts, batch_size=64, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=True)
        else:
            raise ValueError(f"Unknown encoder: {self.mode}")
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = np.asarray(vectors / np.maximum(lengths, 1e-12), dtype=np.float32)
        np.save(path, self.vectors, allow_pickle=False)

    def relevant(self, query, memory, count):
        if count <= 0 or not memory:
            return []
        scores = [(float(self.vectors[query] @ self.vectors[r["item_id"]]), index)
                  for index, r in enumerate(memory)]
        scores.sort(key=lambda pair: (-pair[0], -pair[1]))
        return [index for _, index in scores[:count]]


class ExposureEnvironment:
    def __init__(self, counts, alpha):
        self.weights = np.power(np.asarray(counts, dtype=float), alpha)

    def expose(self, seen, target, size, seed):
        weights = self.weights.copy()
        weights[list(set(seen) | {target})] = 0
        available = np.flatnonzero(weights > 0)
        size = min(size, len(available))
        if size == 0:
            return []
        probability = weights[available] / weights[available].sum()
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.choice(available, size=size, replace=False, p=probability)]


class TargetAwareExposure(ExposureEnvironment):
    """Experimental metadata-only attack variant; not the original exposure protocol.

    Mix popular-item weights with a semantic neighborhood of the target. No
    real-user history, test labels, victim predictions or gradients are used.
    """
    def __init__(self, counts, alpha, vectors, target, pool_size, semantic_fraction):
        super().__init__(counts, alpha)
        similarity = np.asarray(vectors) @ np.asarray(vectors)[target]
        available = np.flatnonzero((np.asarray(counts) > 0) & (np.arange(len(counts)) != target))
        nearest = available[np.lexsort((available, -similarity[available]))[:pool_size]]
        popular = self.weights / max(self.weights.sum(), 1e-12)
        semantic = np.zeros_like(popular)
        if len(nearest):
            values = np.exp((similarity[nearest] - similarity[nearest].max()) / 0.1)
            semantic[nearest] = values / values.sum()
        self.weights = ((1-semantic_fraction) * popular + semantic_fraction * semantic)


def request_validated(client, task, payload, validator, retries, instruction=None):
    last = None
    for attempt in range(retries):
        try:
            prompt = instruction or PROMPTS[task]
            if task == "select" and payload.get("min_count", 0):
                prompt += (" For this dense-history experimental variant, choose at least min_count "
                                "and at most max_count supplied items. This minimum overrides the option to choose none; "
                                "prefer items consistent with your profile.")
            result = client.generate(task, prompt, {**payload, "schema_attempt": attempt})
            return validator(result)
        except (ValueError, KeyError, TypeError) as exc:
            last = exc
    raise ValueError(f"Invalid {task} response after {retries} schema attempts: {last}")


def valid_profile(result):
    profile = result["profile"]
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    if type(profile.get("age")) is not int or not 18 <= profile["age"] <= 100:
        raise ValueError("profile age must be an adult integer")
    if not isinstance(profile.get("occupation"), str) or not profile["occupation"].strip():
        raise ValueError("profile occupation missing")
    if not isinstance(profile.get("rating_habits"), str) or not profile["rating_habits"].strip():
        raise ValueError("rating habits missing")
    if not isinstance(profile.get("interests"), list) or not profile["interests"] or not all(isinstance(x, str) for x in profile["interests"]):
        raise ValueError("interests must be nonempty strings")
    return profile


def valid_ids(result, field, allowed, exact=None, maximum=None, minimum=0):
    ids = result[field]
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ValueError("IDs must be integer arrays")
    if len(set(ids)) != len(ids) or not set(ids) <= set(allowed):
        raise ValueError("Duplicate or out-of-candidate IDs")
    if exact is not None and len(ids) != exact:
        raise ValueError("Incorrect selection count")
    if maximum is not None and len(ids) > maximum:
        raise ValueError("Too many selected items")
    if len(ids) < minimum:
        raise ValueError("Too few selected items")
    return ids


def valid_ratings(result, selected):
    rows = result["ratings"]
    if not isinstance(rows, list):
        raise ValueError("ratings must be a list")
    ids = []
    for row in rows:
        if type(row["item_id"]) is not int or type(row["rating"]) is not int or not 1 <= row["rating"] <= 5:
            raise ValueError("Invalid integer item ID or rating")
        ids.append(row["item_id"])
    if len(set(ids)) != len(ids) or set(ids) != set(selected):
        raise ValueError("Must rate each selected item exactly once")
    return [{"item_id": row["item_id"], "rating": row["rating"]} for row in rows]


def _profile_key(profile):
    return (int(profile["age"]), profile["occupation"].strip().casefold(),
            tuple(sorted(x.strip().casefold() for x in profile["interests"])))


def _interest_jaccard(left, right):
    a = {x.strip().casefold() for x in left["interests"] if x.strip()}
    b = {x.strip().casefold() for x in right["interests"] if x.strip()}
    return len(a & b) / max(len(a | b), 1)


def profile_diversity_report(profiles, existing=()):
    """Deterministic minimum diversity gate; LLM self-claims are not sufficient."""
    count = len(profiles)
    occupations = [p["occupation"].strip().casefold() for p in profiles]
    ages = [int(p["age"]) for p in profiles]
    exact = [_profile_key(p) for p in profiles]
    existing_keys = {_profile_key(p) for p in existing}
    max_jaccard = max((_interest_jaccard(a, b) for i, a in enumerate(profiles)
                        for b in profiles[i + 1:]), default=0.0)
    required_occupations = min(count, max(1, math.ceil(count * 0.5)))
    reasons = []
    if len(set(exact)) != count or any(key in existing_keys for key in exact):
        reasons.append("duplicate profile")
    if len(set(occupations)) < required_occupations:
        reasons.append("too few occupations")
    if count >= 3 and max(ages) - min(ages) < 5:
        reasons.append("age range below 5 years")
    if max_jaccard >= 0.95:
        reasons.append("near-identical interests")
    return {"passed": not reasons, "reasons": reasons,
            "unique_occupations": len(set(occupations)),
            "age_range": max(ages)-min(ages) if ages else 0,
            "max_interest_jaccard": max_jaccard}


def _profile_distance(left, right):
    age = min(abs(int(left["age"])-int(right["age"])) / 20.0, 1.0)
    occupation = float(left["occupation"].strip().casefold() != right["occupation"].strip().casefold())
    return age + occupation + (1.0 - _interest_jaccard(left, right))


def greedy_diverse_ids(candidates, count, existing=()):
    """Deterministic fallback when the LLM's subset is still homogeneous."""
    remaining = list(candidates)
    chosen = []
    while remaining and len(chosen) < count:
        if not chosen and not existing:
            selected = remaining[0]
        else:
            reference = [x["profile"] for x in chosen] + list(existing)
            selected = max(remaining, key=lambda row: (
                min((_profile_distance(row["profile"], other) for other in reference), default=0.0),
                -row["candidate_id"]))
        chosen.append(selected)
        remaining.remove(selected)
    return [row["candidate_id"] for row in chosen]


def make_profiles(client, target, count, settings):
    retries = settings["validation_retries"]
    strict = settings.get("profile_diversity_mode", "legacy") == "strict_v1"
    def analysis_validator(value):
        for field in ("features", "likely_preferences"):
            if not isinstance(value[field], list) or not all(isinstance(x, str) for x in value[field]):
                raise ValueError("analysis fields must be string lists")
        return value
    analysis = request_validated(client, "analyze_target", {"target": target}, analysis_validator, retries)
    result = []
    # Bounded groups avoid putting thousands of profiles into one context.
    for group_start in range(0, count, 16):
        required = min(16, count - group_start)
        candidates = []
        for local in range(math.ceil(required * settings["profile_candidate_multiplier"])):
            candidate_id = group_start * math.ceil(settings["profile_candidate_multiplier"]) + local
            accepted = None
            for regeneration in range(retries):
                payload = {
                    "target": target, "analysis": analysis, "candidate_id": candidate_id,
                    "regeneration": regeneration}
                if strict:
                    payload["existing_profiles"] = [x["profile"] for x in candidates[-8:]] + result[-8:]
                profile = request_validated(client, "profile", payload, valid_profile, retries,
                                            STRICT_PROFILE_PROMPT if strict else None)
                def preference_validator(value):
                    if type(value["prefers_target"]) is not bool:
                        raise ValueError("prefers_target must be boolean")
                    return value["prefers_target"]
                prefers = request_validated(client, "validate_profile", {"target": target, "profile": profile},
                                             preference_validator, retries)
                if prefers:
                    accepted = profile
                    break
            if accepted is not None:
                candidates.append({"candidate_id": candidate_id, "profile": accepted})
        if len(candidates) < required:
            raise ValueError("Too few validated profiles; no fallback to random users. Review cached responses.")
        chosen = request_validated(client, "diversity", {"count": required, "candidates": candidates},
            lambda value: valid_ids(value, "selected_ids", [x["candidate_id"] for x in candidates], exact=required), retries)
        lookup = {x["candidate_id"]: x["profile"] for x in candidates}
        if strict and not profile_diversity_report([lookup[i] for i in chosen], result)["passed"]:
            chosen = greedy_diverse_ids(candidates, required, result)
            report = profile_diversity_report([lookup[i] for i in chosen], result)
            if not report["passed"]:
                raise ValueError(f"Strict profile diversity failed: {report['reasons']}")
        result.extend(lookup[i] for i in chosen)
        print(f"Profiles ready: {len(result)}/{count} (validated and diversified)", flush=True)
    return result


class FakeUserAgent:
    def __init__(self, user_id, profile, memory, client, encoder, settings):
        self.user_id, self.profile, self.memory = user_id, profile, memory
        self.client, self.encoder, self.settings = client, encoder, settings

    def retrieve(self, item):
        if self.settings["ablation"] == "no_memory":
            return []
        indices = set(self.encoder.relevant(item, self.memory, self.settings["relevant_k"]))
        recent = self.settings["recent_m"]
        if recent:
            indices.update(range(max(0, len(self.memory) - recent), len(self.memory)))
        return [self.memory[i] for i in sorted(indices)]

    def act(self, candidates, items, round_id, remaining):
        limit = min(remaining, self.settings["max_selected_per_round"], len(candidates))
        if not candidates or limit <= 0:
            return []
        retries = self.settings["validation_retries"]
        minimum = min(limit, self.settings.get("min_selected_per_round", 0))
        payload = {
            "agent_id": self.user_id, "round": round_id, "profile": self.profile,
            "reference_count": min(limit, self.settings["reference_per_round"]),
            "max_count": limit, "candidates": [public_item(items[i]) for i in candidates]}
        if minimum:
            payload["min_count"] = minimum
        selected = request_validated(self.client, "select", payload,
            lambda value: valid_ids(value, "item_ids", candidates, maximum=limit, minimum=minimum), retries)
        if not selected:
            return []
        # Retrieve per queried item. Same-round ratings are generated as a batch.
        memories = {str(i): [{**m, "title": items[m["item_id"]]["title"][:500]}
                              for m in self.retrieve(i)] for i in selected}
        ratings = request_validated(self.client, "rate", {
            "agent_id": self.user_id, "round": round_id, "profile": self.profile,
            "memories_by_item": memories, "items": [public_item(items[i]) for i in selected]},
            lambda value: valid_ratings(value, selected), retries)
        return ratings


def agent_directory(config, target, mock=False):
    mode = "mock" if mock else "real"
    return resolve(config, config["agent"]["output"]) / mode / f"target_{target}"


def generate_agent_attack(config, dataset, target, mock=False):
    settings = config["agent"]
    if settings["ablation"] not in {"full", "no_memory", "no_profile"}:
        raise ValueError("Unknown Agent ablation")
    if target not in dataset["targets"]:
        raise ValueError("Target is not in the frozen target list.")
    directory = agent_directory(config, target, mock)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".generation.lock"
    try:
        lock_handle = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError("Generation already running or interrupted lock remains. Check process before removing lock.") from exc
    try:
        return _generate(config, dataset, target, mock, directory)
    finally:
        lock_handle.close()
        lock.unlink(missing_ok=True)


def _generate(config, dataset, target, mock, directory):
    settings = config["agent"]
    count = fake_user_count(dataset["manifest"]["users"], config["attack"])
    client = MockClient() if mock else ChatClient(config["llm"], directory / "llm")
    prompt_identity = active_prompt_identity(settings)
    signature = digest({"dataset": dataset["manifest"]["dataset_id"], "settings": settings,
                        "fake_count": count, "target": target, "mock": mock,
                        "client": client.identity, "prompts": prompt_identity})
    identity_path = directory / "identity.json"
    if identity_path.exists() and read_json(identity_path)["signature"] != signature:
        raise ValueError("Agent settings/model changed. Choose a new agent.output directory.")
    write_json(identity_path, {"signature": signature, "client": client.identity,
                               "settings": settings, "mock": mock, "prompts": prompt_identity})
    encoder = ItemEncoder(dataset["items"], settings, directory, mock)
    profile_path = directory / "profiles.json"
    if profile_path.exists():
        profiles = read_json(profile_path)
    elif settings["ablation"] == "no_profile":
        profiles = [{"age": 20 + i % 60, "occupation": "unspecified",
                     "interests": ["general pet products"], "rating_habits": "occasional ratings"}
                    for i in range(count)]
        write_json(profile_path, profiles)
    else:
        print(f"Building profiles: {count} fake users; cached responses will be reused.", flush=True)
        profiles = make_profiles(client, public_item(dataset["items"][target]), count, settings)
        write_json(profile_path, profiles)
    real_users, item_count = dataset["manifest"]["users"], dataset["manifest"]["items"]
    counts, _, _ = statistics(dataset["train"], item_count)
    if settings.get("exposure_policy", "popularity") == "target_aware":
        environment = TargetAwareExposure(counts, settings["popularity_alpha"], encoder.vectors,
            target, settings.get("semantic_pool_size", 200), settings.get("semantic_fraction", 0.5))
    else:
        environment = ExposureEnvironment(counts, settings["popularity_alpha"])
    all_rows, lengths = [], []
    print(f"Generating ratings: {count} fake users, {settings['rounds']} rounds each.", flush=True)
    for index, profile in enumerate(profiles):
        user = real_users + index
        state_path = directory / "states" / f"{user}.json"
        if state_path.exists():
            state = read_json(state_path)
            if state.get("signature") != signature or state.get("profile_hash") != digest(profile):
                raise ValueError("Agent checkpoint identity/profile changed; use a new output directory.")
        else:
            state = {"next_round": 0, "memory": [{"item_id": target, "rating": 5, "round": -1}]}
        agent = FakeUserAgent(user, profile, state["memory"], client, encoder, settings)
        for round_id in range(state["next_round"], settings["rounds"]):
            seen = {r["item_id"] for r in agent.memory}
            remaining = settings["filler_cap"] - len(seen - {target})
            exposed = environment.expose(seen, target, settings["exposure_size"],
                stable_seed(settings["seed"], target, user, round_id))
            ratings = agent.act(exposed, dataset["items"], round_id, remaining)
            agent.memory.extend({**r, "round": round_id} for r in ratings)
            state = {"next_round": round_id + 1, "memory": agent.memory,
                     "signature": signature, "profile_hash": digest(profile)}
            write_json(state_path, state)
            append_jsonl(directory / "actions.jsonl", {"user_id": user, "round": round_id,
                                                       "exposed": exposed, "ratings": ratings})
        rows = [{"user_id": user, "item_id": r["item_id"], "rating": r["rating"]} for r in agent.memory]
        all_rows.extend(rows)
        lengths.append(len(rows) - 1)
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"{'MOCK' if mock else 'LLM'} target {target}: {index + 1}/{count} users", flush=True)
    validate_fake(all_rows, real_users, item_count, target, count)
    path = directory / "fake_ratings.csv"
    save_attack(path, all_rows)
    write_json(directory / "manifest.json", {
        "signature": signature, "dataset_id": dataset["manifest"]["dataset_id"], "target": target,
        "mock": mock, "ablation": settings["ablation"], "fake_users": count,
        "variant": settings.get("variant", "rating_original"),
        "exposure_policy": settings.get("exposure_policy", "popularity"),
        "min_selected_per_round": settings.get("min_selected_per_round", 0),
        "generation_settings": settings,
        "interactions": len(all_rows), "filler_lengths": lengths, "encoder": encoder.identity,
        "client": client.identity, "sha256": file_hash(path),
        "scope": "rating-only; no generated reviews or target-feature propagation",
        "profiles_sha256": file_hash(profile_path),
    })
    print(f"Saved {'MOCK FIXTURE' if mock else 'LLM'} attack: {path}")
    return path
