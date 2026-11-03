"""Run `python -m agentsa --help` from the project directory."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

from .utils import load_config, read_json, resolve


def csv_values(value, convert=str):
    return [convert(x.strip()) for x in value.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="AgentSA rating and review-attack research implementation")
    parser.add_argument("--config", default="configs/phase1.yaml", help="YAML config (global argument before subcommand)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check environment; never contacts a model service")
    commands.add_parser("api-check", help="Send a tiny real JSON request; uses API credits, no dataset/encoder needed")
    commands.add_parser("prepare", help="Prepare data, save splits/targets/hashes, raw archives unchanged")
    commands.add_parser("prepare-reviews", help="Prepare review-text sidecar aligned to frozen rating splits")
    commands.add_parser("estimate", help="Estimate attack size and request count; no model calls")
    commands.add_parser("estimate-reviews", help="Estimate post-hoc review requests; no model calls")
    clean = commands.add_parser("train-clean", help="Train clean victim models only")
    clean.add_argument("--models")
    clean.add_argument("--seeds")
    agent = commands.add_parser("generate-agent", help="Generate ONE target by default; real mode may incur API costs")
    agent.add_argument("--target", type=int)
    agent.add_argument("--all-targets", action="store_true")
    agent.add_argument("--mock", action="store_true", help="Protocol fixture only: NO real LLM calls")
    reviews = commands.add_parser("generate-reviews", help="Generate plain and target-feature-propagated reviews")
    reviews.add_argument("--target", type=int)
    reviews.add_argument("--mock", action="store_true", help="Deterministic fixture; NO real LLM calls")
    audit = commands.add_parser("audit-reviews", help="Offline quality audit for generated plain/TFP reviews")
    audit.add_argument("--target", type=int)
    audit.add_argument("--mock", action="store_true")
    review_run = commands.add_parser("run-reviews", help="Train/evaluate review-aware victim models")
    review_run.add_argument("--models")
    review_run.add_argument("--seeds")
    review_run.add_argument("--target-count", type=int)
    review_run.add_argument("--mock-agent", action="store_true")
    run = commands.add_parser("run", help="Train/evaluate attacks; Agent CSVs must already exist")
    run.add_argument("--models")
    run.add_argument("--methods")
    run.add_argument("--seeds")
    run.add_argument("--target-count", type=int)
    run.add_argument("--include-agent", action="store_true")
    run.add_argument("--match-agent-budget", action="store_true")
    run.add_argument("--mock-agent", action="store_true", help="Isolate all output under mock_validation")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        import torch
        from .llm import connection_info
        connection = connection_info(config["llm"])
        print(json.dumps({"python": sys.executable, "python_version": sys.version.split()[0],
            "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
            "semantic_dependency_installed": importlib.util.find_spec("sentence_transformers") is not None,
            "llm_provider": connection["provider"], "llm_model": connection["model"],
            "model_environment_configured": bool(connection["model"]),
            "endpoint_environment_configured": bool(connection["base_url"]),
            "credential_present": connection["credential_present"],
            "credential_environment_name": connection["api_key_env"],
            "raw_files": {k: resolve(config, config["data"][k]).is_file() for k in ("reviews", "metadata")},
            "data_scope": "full dataset" if config["data"]["year"] is None else config["data"]["year"]}, indent=2))
        return
    if args.command == "api-check":
        from .llm import ChatClient
        directory = resolve(config, config["agent"]["output"]) / "api_checks"
        client = ChatClient(config["llm"], directory)
        # Unique nonce ensures a repeated check tests the current credentials/service.
        result = client.generate("connection_check", 'Return JSON exactly like this example: {"ok": true}.',
                                 {"check_id": time.time_ns()})
        if result.get("ok") is not True:
            raise ValueError("Connection returned JSON but not the expected {ok: true} result.")
        print(json.dumps({"ok": True, "provider": client.identity["provider"],
                          "model": client.model, "usage_file": str(client.ledger)}, indent=2))
        return
    from .data import load_dataset, prepare
    if args.command == "prepare":
        manifest = prepare(config)
        print(json.dumps({k: manifest[k] for k in ("users", "items", "interactions", "split_sizes", "items_missing_title", "targets")}, indent=2))
        return
    dataset = load_dataset(config)
    if args.command == "prepare-reviews":
        from .review_data import prepare_review_data
        manifest = prepare_review_data(config, dataset)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "estimate":
        users = dataset["manifest"]["users"]
        count = math.floor(users * config["attack"]["fake_ratio"])
        a = config["agent"]
        # Two calls per candidate (profile + validation), plus diversity group selection.
        profile_calls = 1 + 2 * math.ceil(count * a["profile_candidate_multiplier"]) + math.ceil(count / 16)
        interaction_calls = count * a["rounds"] * 2
        print(json.dumps({"real_users": users, "fake_users_per_target": count,
            "targets": len(dataset["targets"]), "baseline_filler_per_user": config["attack"]["filler_count"],
            "approx_requests_per_target": profile_calls + interaction_calls,
            "approx_requests_all_targets": (profile_calls + interaction_calls) * len(dataset["targets"]),
            "note": "Rating-stage estimate assumes nonempty selections, no retries; no model/API has been called.",
            "per_target_request_limit": config["llm"]["max_requests"]}, indent=2))
        return
    if args.command == "estimate-reviews":
        users = dataset["manifest"]["users"]
        count = math.floor(users * config["attack"]["fake_ratio"])
        rounds = config["agent"]["rounds"]
        strict_v3 = config.get("reviews", {}).get("protocol") == "strict_v3"
        calls_per_round = 3 if strict_v3 else 2
        minimum_per_target = 1 + count * (1 + 2 * rounds)
        maximum_per_target = 1 + count * (1 + calls_per_round * rounds)
        print(json.dumps({"real_users": users, "fake_users_per_target": count,
            "rounds": rounds, "target_analysis_calls": 1,
            "target_review_calls": count, "plain_review_batch_calls": count * rounds,
            "feature_extraction_calls": count * rounds if strict_v3 else 0,
            "rewrite_calls_maximum": count * rounds if strict_v3 else 0,
            "propagation_batch_calls": 0 if strict_v3 else count * rounds,
            "minimum_requests_per_target": minimum_per_target,
            "maximum_requests_per_target": maximum_per_target,
            "approx_requests_per_target": maximum_per_target,
            "note": "No retries; strict_v3 rewrite calls occur only when approved features remain. The review stage has its own cache and usage ledger."}, indent=2))
        return
    if args.command == "generate-agent":
        if args.target is not None and args.all_targets:
            parser.error("Choose --target or --all-targets, not both")
        from .agents import generate_agent_attack
        targets = dataset["targets"] if args.all_targets else [args.target if args.target is not None else dataset["targets"][0]]
        for target in targets:
            generate_agent_attack(config, dataset, target, args.mock)
        return
    if args.command == "generate-reviews":
        from .review_generation import generate_review_attack
        target = args.target if args.target is not None else dataset["targets"][0]
        generate_review_attack(config, dataset, target, args.mock)
        return
    if args.command == "audit-reviews":
        from .review_audit import audit_review_attack
        target = args.target if args.target is not None else dataset["targets"][0]
        print(json.dumps(audit_review_attack(config, dataset, target, args.mock),
                         ensure_ascii=False, indent=2))
        return
    if args.command == "run-reviews":
        from .review_experiment import run_review_experiments
        models = csv_values(args.models) if args.models else ["dual_tower_early", "dual_tower_late"]
        seeds = csv_values(args.seeds, int) if args.seeds else config["experiment"]["seeds"]
        if set(models) - {"dual_tower_early", "dual_tower_late"} or not seeds:
            parser.error("Review models must be dual_tower_early/dual_tower_late with at least one seed")
        count = args.target_count if args.target_count is not None else len(dataset["targets"])
        if not 1 <= count <= len(dataset["targets"]):
            parser.error("--target-count must fit the saved target list")
        run_review_experiments(config, dataset, models, seeds, dataset["targets"][:count], args.mock_agent)
        return
    from .experiment import run_experiments, train_clean
    models = csv_values(args.models) if args.models else config["experiment"]["models"]
    seeds = csv_values(args.seeds, int) if args.seeds else config["experiment"]["seeds"]
    if not models or set(models) - {"nmf", "neunmf"} or not seeds:
        parser.error("Provide models nmf/neunmf and at least one seed")
    if args.command == "train-clean":
        train_clean(config, dataset, models, seeds)
        return
    methods = csv_values(args.methods) if args.methods else config["experiment"]["methods"]
    if args.target_count is not None and not 1 <= args.target_count <= len(dataset["targets"]):
        parser.error("--target-count must fit the saved target list")
    targets = dataset["targets"][:args.target_count] if args.target_count else dataset["targets"]
    run_experiments(config, dataset, models, methods, seeds, targets,
                    args.include_agent, args.match_agent_budget, args.mock_agent)


if __name__ == "__main__":
    main()
