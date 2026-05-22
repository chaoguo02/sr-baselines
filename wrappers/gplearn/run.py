import argparse
import os
import subprocess
import sys

import yaml


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WRAPPER_CONFIG = os.path.join("wrappers", "gplearn", "config.yml")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper entrypoint for the gplearn baseline.")
    parser.add_argument(
        "--wrapper-config",
        default=DEFAULT_WRAPPER_CONFIG,
        help="Path to wrapper config YAML.",
    )
    parser.add_argument("--function-id", type=int, default=None, help="Run one benchmark function.")
    parser.add_argument("--all", action="store_true", help="Run all benchmark functions.")
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper_config = load_yaml(resolve_path(args.wrapper_config))
    legacy_config_path = wrapper_config["legacy_config_path"]
    legacy_script = resolve_path(os.path.join("gplearn", "run_gplearn_benchmark.py"))

    cmd = [
        sys.executable,
        legacy_script,
        "--config",
        legacy_config_path,
    ]
    if args.all:
        cmd.append("--all")
    elif args.function_id is not None:
        cmd.extend(["--function-id", str(args.function_id)])
    else:
        raise ValueError("Please provide --function-id N or use --all.")

    completed = subprocess.run(cmd, cwd=ROOT_DIR)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
