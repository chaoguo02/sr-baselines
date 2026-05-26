"""Run all baselines across multiple benchmark functions with multiple independent runs.

Usage:
    python run_all_baselines.py                           # all baselines, func1-10, 10 runs each
    python run_all_baselines.py --baselines gplearn pysr  # only gplearn + pysr
    python run_all_baselines.py --functions 1 2 3         # only func1-3
    python run_all_baselines.py --runs 5                  # 5 runs per (baseline, func)
    python run_all_baselines.py --resume                  # skip completed runs
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback

import yaml

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(row["function_id"]): row for row in rows}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

BASELINES = {}


def register(name, import_module, train_one_name, config_path, seed_key,
             align_overrides=None, setup_fn=None, uses_bracket_access=False,
             no_seed_warning=False):
    """Register a baseline for the multi-run launcher.

    Parameters
    ----------
    name : str                e.g. "gplearn"
    import_module : str       fully qualified module path, e.g. "gplearn.run_gplearn_benchmark"
    train_one_name : str      function name inside module, e.g. "train_one"
    config_path : str         relative path from ROOT_DIR to default config YAML
    seed_key : str or None    config dict key for seed, or None if unsupported
    align_overrides : dict    config overrides to align with main experiment
    setup_fn : callable or None   optional function(config) called before each run
    uses_bracket_access : bool    if True, use config["key"] instead of config.get("key")
    no_seed_warning : bool    if True, warn at import that seeding is not supported
    """
    BASELINES[name] = {
        "import_module": import_module,
        "train_one_name": train_one_name,
        "config_path": config_path,
        "seed_key": seed_key,
        "align_overrides": align_overrides or {},
        "setup_fn": setup_fn,
        "uses_bracket_access": uses_bracket_access,
        "no_seed_warning": no_seed_warning,
    }


def _resolve_pysr_env(config):
    """Preprare Julia runtime for PySR before calling train_one."""
    # Deferred import to avoid julia deps until needed
    from wrappers.pysr.run import prepare_runtime_env
    prepare_runtime_env(config)


# --- Register all 9 baselines ---

register(
    name="gplearn",
    import_module="run_gplearn_benchmark",   # run_gplearn_benchmark.py in gplearn/ subdir
    train_one_name="train_one",
    config_path=os.path.join("gplearn", "config_gplearn.yml"),
    seed_key="seed",
    uses_bracket_access=True,
    align_overrides={},
)

register(
    name="icsr",
    import_module="wrappers.icsr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "icsr", "config.yml"),
    seed_key="seed",
    align_overrides={},
)

register(
    name="llm_sr",
    import_module="wrappers.llm_sr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "llm_sr", "config.yml"),
    seed_key=None,
    no_seed_warning=True,
    align_overrides={},
)

register(
    name="sr_llm",
    import_module="wrappers.sr_llm.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "sr_llm", "config.yml"),
    seed_key="seed",
    align_overrides={},
)

register(
    name="lasr",
    import_module="wrappers.lasr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "lasr", "config.yml"),
    seed_key="seed",
    align_overrides={},
)

register(
    name="pysr",
    import_module="wrappers.pysr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "pysr", "config.yml"),
    seed_key="seed",
    align_overrides={"population_size": 500},
    setup_fn=_resolve_pysr_env,
)

register(
    name="gp_gomea",
    import_module="wrappers.gp_gomea.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "gp_gomea", "config.yml"),
    seed_key="seed",
    align_overrides={},
)

register(
    name="dsr",
    import_module="wrappers.dsr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "dsr", "config.yml"),
    seed_key="seed",
    align_overrides={},
)

register(
    name="udsr",
    import_module="wrappers.dsr.run",
    train_one_name="train_one",
    config_path=os.path.join("wrappers", "dsr", "config_udsr.yml"),
    seed_key="seed",
    align_overrides={},
)


def import_train_one(info):
    """Dynamically import and return (train_one_fn, metadata_loader, yaml_loader, path_resolver)."""
    mod_name = info["import_module"]
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    # Special path for gplearn (legacy script in gplearn/ subdirectory, no __init__.py)
    if info["name"] == "gplearn":
        gplearn_dir = os.path.join(ROOT_DIR, "gplearn")
        if gplearn_dir not in sys.path:
            sys.path.insert(0, gplearn_dir)

    __import__(mod_name)
    mod = sys.modules[mod_name]

    # Ensure ROOT_DIR is first on sys.path for downstream imports
    if ROOT_DIR in sys.path:
        sys.path.remove(ROOT_DIR)
    sys.path.insert(0, ROOT_DIR)
    train_one = getattr(mod, info["train_one_name"])
    return train_one, mod


def make_run_config(info, base_config, function_id, run_index, base_seed, results_root):
    """Produce a deep-copied config dict for one specific (baseline, func, run)."""
    config = copy.deepcopy(base_config)

    # Override OUTPUT_PATH — set to run level; each wrapper's train_one appends
    # func{function_id}/ internally, resulting in:
    #   results/<bl_name>/run_<run_index>/func<N>/run_summary.json
    run_output = f"{info['name']}/run_{run_index}"
    config["OUTPUT_PATH"] = os.path.join(results_root, run_output).replace("\\", "/")

    # Override seed
    if info["seed_key"] is not None:
        seed_value = base_seed + run_index
        if info["uses_bracket_access"]:
            # gplearn legacy uses bracket syntax, must assign directly
            config[info["seed_key"]] = seed_value
        else:
            config[info["seed_key"]] = seed_value

    # Apply parameter alignment overrides
    for k, v in info["align_overrides"].items():
        if info["uses_bracket_access"]:
            config[k] = v
        else:
            config[k] = v

    return config


def run_single(info, train_one, config, function_id, metadata, run_index):
    """Run one baseline on one function and return the result dict (or error dict)."""
    start = time.time()
    try:
        result = train_one(
            function_id=function_id,
            metadata=metadata,
            config=config,
        )
        elapsed = time.time() - start
        if isinstance(result, dict):
            result["run_index"] = run_index
            result["run_seed"] = config.get(info["seed_key"], "N/A") if info["seed_key"] else "N/A"
        return result
    except Exception as exc:
        elapsed = time.time() - start
        return {
            "algorithm": info["name"],
            "function_id": function_id,
            "run_index": run_index,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": elapsed,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-run baseline launcher for symbolic regression benchmarks."
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        choices=list(BASELINES.keys()),
        help="Baselines to run (default: all)",
    )
    parser.add_argument(
        "--functions", nargs="+", type=int, default=None,
        help="Function IDs to run (default: 1..10)",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of independent runs per (baseline, function) pair (default: 10)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base random seed for run 0; each subsequent run adds 1 (default: 42)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs that already have run_summary.json",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Root directory for all baseline results (default: results/)",
    )
    args = parser.parse_args()

    baseline_names = args.baselines or sorted(BASELINES.keys())
    function_ids = args.functions or list(range(1, 11))
    num_runs = args.runs
    base_seed = args.base_seed
    results_root = resolve_path(args.results_dir)
    summary_path = os.path.join(results_root, "all_baselines_summary.jsonl")
    ensure_dir(results_root)

    # Load benchmark metadata once
    metadata_config = load_yaml(
        resolve_path(BASELINES[baseline_names[0]]["config_path"])
    )
    metadata = load_metadata(resolve_path(metadata_config["BENCHMARK_METADATA_PATH"]))

    all_results = []
    summary = {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    for bl_name in baseline_names:
        info = BASELINES[bl_name]
        info["name"] = bl_name  # store name for easy access

        print(f"\n{'=' * 60}")
        print(f"[{bl_name}] Initializing...")
        print(f"{'=' * 60}")

        # Import module
        try:
            train_one, mod = import_train_one(info)
        except Exception as exc:
            print(f"[{bl_name}] IMPORT FAILED: {exc}")
            summary["failed"] += num_runs * len(function_ids)
            continue

        # Load config
        try:
            base_config = load_yaml(resolve_path(info["config_path"]))
        except Exception as exc:
            print(f"[{bl_name}] CONFIG LOAD FAILED: {exc}")
            summary["failed"] += num_runs * len(function_ids)
            continue

        # Warn if no seed support
        if info["no_seed_warning"]:
            print(f"[{bl_name}] WARNING: This baseline does not support random seed control. "
                  f"Each run will use the algorithm's internal default.")

        # Setup function (e.g. PySR runtime env)
        if info["setup_fn"]:
            try:
                info["setup_fn"](base_config)
            except Exception as exc:
                print(f"[{bl_name}] SETUP FAILED: {exc}")
                summary["failed"] += num_runs * len(function_ids)
                continue

        # Ensure metadata has the needed function
        available = sorted(metadata.keys())
        run_funcs = [fid for fid in function_ids if fid in metadata]
        if not run_funcs:
            print(f"[{bl_name}] No matching function IDs in metadata (available: {available})")
            summary["failed"] += num_runs * len(function_ids)
            continue

        for function_id in run_funcs:
            for run_index in range(num_runs):
                # Determine output directory: wrapper appends func{N}/ internally
                run_output_dir = os.path.join(
                    results_root, f"{bl_name}/run_{run_index}"
                )
                func_run_dir = os.path.join(run_output_dir, f"func{function_id}")
                summary_path_check = os.path.join(func_run_dir, "run_summary.json")
                if args.resume and os.path.exists(summary_path_check):
                    print(f"  [{bl_name}] func{function_id} run_{run_index} → SKIPPED (exists)")
                    summary["skipped"] += 1
                    # Load existing result for the summary
                    try:
                        with open(summary_path_check, "r", encoding="utf-8") as _f:
                            existing = json.load(_f)
                        if isinstance(existing, dict):
                            existing["baseline"] = bl_name
                            existing["run_index"] = run_index
                            all_results.append(existing)
                    except Exception:
                        pass
                    continue

                # Build config for this run
                run_cfg = make_run_config(
                    info, base_config, function_id, run_index,
                    base_seed, args.results_dir,
                )

                # Ensure output dir exists
                ensure_dir(run_output_dir)

                # Run
                label = f"[{bl_name}] func{function_id} run_{run_index}"
                print(f"  {label} (seed={run_cfg.get(info['seed_key'], 'N/A') if info['seed_key'] else 'N/A'}) ...", end=" ")
                sys.stdout.flush()

                result = run_single(info, train_one, run_cfg, function_id, metadata, run_index)

                if result.get("status") == "failed":
                    print("FAILED")
                    error_path = os.path.join(func_run_dir, "run_error.json")
                    with open(error_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    summary["failed"] += 1
                else:
                    train_mse = result.get("train_mse", "?")
                    test_mse = result.get("test_mse", "?")
                    rt = result.get("runtime_seconds", 0)
                    print(f"OK  train_mse={train_mse:.4e}  test_mse={test_mse:.4e}  time={rt:.1f}s")
                    summary["succeeded"] += 1

                result["baseline"] = bl_name
                all_results.append(result)
                summary["total"] += 1

    # Write unified summary
    print(f"\n{'=' * 60}")
    print(f"Summary: {summary['succeeded']} succeeded, "
          f"{summary['failed']} failed, "
          f"{summary['skipped']} skipped "
          f"(total {summary['total']} runs)")
    print(f"{'=' * 60}")

    with open(summary_path, "w", encoding="utf-8") as f:
        for entry in all_results:
            # Strip traceback from summary to keep it readable
            entry_summary = {k: v for k, v in entry.items() if k != "traceback"}
            f.write(json.dumps(entry_summary, ensure_ascii=False) + "\n")
    print(f"\nUnified summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
