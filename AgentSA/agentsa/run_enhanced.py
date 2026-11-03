"""Run the authorized single-target enhanced-Agent generation and comparison.

Credentials stay in process environment only. Use --check-only for offline checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import subprocess
import sys

from .data import load_dataset
from .experiment import checked_agent_data
from .utils import load_config, resolve, write_json


def completed_generation(config, dataset, target):
    directory = resolve(config, config["agent"]["output"]) / "real" / f"target_{target}"
    if not (directory / "manifest.json").exists():
        return False
    # Invalid completed artifacts must raise rather than trigger paid regeneration.
    checked_agent_data(config, dataset, target, False)
    return True


def workflow(config_path, check_only=False):
    config = load_config(config_path)
    if config["agent"].get("variant") != "target_aware_dense_v1":
        raise ValueError("This launcher requires the target_aware_dense_v1 configuration.")
    dataset = load_dataset(config)
    target = dataset["targets"][0]
    complete = completed_generation(config, dataset, target)
    key_name = config["llm"]["api_key_env"]
    status_path = resolve(config, config["experiment"]["output"]) / "workflow_status.json"
    summary = {"target": target, "fake_users": int(dataset["manifest"]["users"] * config["attack"]["fake_ratio"]),
               "generation_complete": complete, "credential_present": bool(os.environ.get(key_name)),
               "training_seed": 11, "models": ["nmf", "neunmf"],
               "experiment_directory": str(resolve(config, config["experiment"]["output"]) / "experiments"),
               "scope": "one frozen target; real generation followed by matched-budget comparison"}
    if check_only:
        return summary
    generation_dir = resolve(config, config["agent"]["output"]) / "real" / f"target_{target}"
    if not complete and (generation_dir / ".generation.lock").exists():
        raise RuntimeError("Generation lock exists. Check the old process before removing it; no requests sent.")
    previous_key = os.environ.get(key_name)
    if not complete and not previous_key:
        if not sys.stdin.isatty():
            raise RuntimeError(f"{key_name} is missing. Run this launcher in your local terminal to enter it securely.")
        secret = getpass.getpass("DeepSeek API Key (hidden; not saved): ").strip()
        if not secret:
            raise ValueError("Empty API Key; no requests sent.")
        os.environ[key_name] = secret
        del secret
    state = {**summary, "pid": os.getpid(), "completed_steps": [], "status": "running"}
    command = [sys.executable, "-m", "agentsa", "--config", config["_config_path"]]

    def run_step(name, arguments):
        state.update(stage=name, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, state)
        print(f"Starting {name}", flush=True)
        subprocess.run(command + arguments, cwd=config["project_root"], check=True)
        state["completed_steps"].append(name)
        write_json(status_path, state)

    try:
        if not complete:
            run_step("api_check", ["api-check"])
            run_step("generation", ["generate-agent", "--target", str(target)])
            if not completed_generation(config, dataset, target):
                raise RuntimeError("Generation did not produce a valid completion manifest; training cancelled.")
        else:
            state["completed_steps"].append("generation_reused")
        run_step("training_and_evaluation", ["run", "--models", "nmf,neunmf", "--seeds", "11",
                "--target-count", "1", "--include-agent", "--match-agent-budget"])
        state.update(status="complete", stage="complete", updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, state)
        return state
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error_type=type(exc).__name__, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, state)
        raise
    finally:
        if previous_key is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = previous_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/formal_target_aware.yaml")
    parser.add_argument("--check-only", action="store_true", help="Offline readiness check; no API calls or training")
    args = parser.parse_args()
    import json
    result = workflow(args.config, args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
