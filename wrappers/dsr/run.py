"""DSO (Deep Symbolic Optimization) wrapper — supports DSR + uDSR modes."""

import argparse
import json
import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.env_loader import load_env_file
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path


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


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_xy(dataset_root, function_id, n_variables):
    train_path = os.path.join(dataset_root, f"fitness_cases{function_id}.csv")
    test_path = os.path.join(dataset_root, f"hold_out{function_id}.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_columns = [f"x{i+1}" for i in range(int(n_variables))]

    x_train = df_train[feature_columns].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    x_test = df_test[feature_columns].to_numpy(dtype=float)
    y_test = df_test["y"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, train_path, test_path


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def build_dso_config(wrapper_cfg, function_id, meta, output_dir):
    """Build a DSO config dict from wrapper config and benchmark metadata."""
    n_vars = int(meta["n_variables"])

    # Determine DSR vs uDSR mode
    mode = str(wrapper_cfg.get("mode", "dsr")).lower()
    function_set = list(wrapper_cfg.get("function_set", [
        "add", "sub", "mul", "div", "sin", "cos", "exp", "log"
    ]))
    if mode == "udsr" and "poly" not in function_set:
        function_set.append("poly")

    # Map our metric to DSO's
    dso_metric = str(wrapper_cfg.get("dso_metric", "inv_nrmse"))
    dso_metric_params = list(wrapper_cfg.get("dso_metric_params", [1.0]))

    config = {
        "experiment": {
            "seed": int(wrapper_cfg.get("seed", 42)),
            "logdir": output_dir,
        },
        "task": {
            "task_type": "regression",
            "function_set": function_set,
            "metric": dso_metric,
            "metric_params": dso_metric_params,
            "protected": bool(wrapper_cfg.get("protected", False)),
            "threshold": float(wrapper_cfg.get("threshold", 1e-12)),
            "decision_tree_threshold_set": [],
        },
        "training": {
            "n_samples": int(wrapper_cfg.get("n_samples", 2000000)),
            "batch_size": int(wrapper_cfg.get("batch_size", 1000)),
            "epsilon": float(wrapper_cfg.get("epsilon", 0.05)),
            "n_cores_batch": int(wrapper_cfg.get("n_cores_batch", 1)),
            "verbose": bool(wrapper_cfg.get("dso_verbose", True)),
            "early_stopping": bool(wrapper_cfg.get("early_stopping", True)),
        },
        "policy_optimizer": {
            "learning_rate": float(wrapper_cfg.get("learning_rate", 0.0005)),
            "entropy_weight": float(wrapper_cfg.get("entropy_weight", 0.03)),
            "entropy_gamma": float(wrapper_cfg.get("entropy_gamma", 0.7)),
        },
        "policy": {
            "max_length": int(wrapper_cfg.get("max_length", 64)),
        },
        "prior": {
            "length": {
                "min_": int(wrapper_cfg.get("prior_length_min", 4)),
                "max_": int(wrapper_cfg.get("prior_length_max", 64)),
                "on": True,
            },
            "repeat": {
                "tokens": "const",
                "min_": None,
                "max_": int(wrapper_cfg.get("const_max_repeat", 3)),
                "on": True,
            },
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {
                "loc": int(wrapper_cfg.get("soft_length_loc", 10)),
                "scale": int(wrapper_cfg.get("soft_length_scale", 5)),
                "on": True,
            },
            "domain_range": {"on": False},
        },
        "gp_meld": {
            "run_gp_meld": bool(wrapper_cfg.get("run_gp_meld", False)),
        },
        "logging": {
            "save_all_iterations": False,
            "save_summary": False,
            "save_pareto_front": False,
            "save_cache": False,
            "hof": int(wrapper_cfg.get("hof_size", 1)),
        },
    }

    # Add poly optimizer params if in uDSR mode
    if mode == "udsr":
        config["task"]["poly_optimizer_params"] = {
            "degree": int(wrapper_cfg.get("poly_degree", 3)),
            "coef_tol": float(wrapper_cfg.get("poly_coef_tol", 1e-6)),
            "regressor": str(wrapper_cfg.get("poly_regressor", "dso_least_squares")),
            "regressor_params": {
                "cutoff_p_value": float(wrapper_cfg.get("poly_cutoff_p", 1.0)),
                "n_max_terms": wrapper_cfg.get("poly_n_max_terms"),
                "coef_tol": float(wrapper_cfg.get("poly_coef_tol", 1e-6)),
            },
        }

    return config


def train_one(function_id, metadata, wrapper_cfg):
    """Train DSO on one benchmark function and return results in common schema."""
    load_env_file()

    dataset_root = resolve_path(wrapper_cfg["DATASET_PATH"])
    output_root = resolve_path(wrapper_cfg["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    ensure_output_dir(func_output_dir)

    meta = metadata[function_id]
    n_variables = int(meta["n_variables"])
    x_train, y_train, x_test, y_test, train_path, test_path = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=n_variables,
    )

    # Build DSO config
    dso_config = build_dso_config(
        wrapper_cfg=wrapper_cfg,
        function_id=function_id,
        meta=meta,
        output_dir=func_output_dir,
    )

    # Save the generated config for reproducibility
    config_path = os.path.join(func_output_dir, "dso_config.generated.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dso_config, f, indent=2)

    # Import and train DSO
    from dso.task.regression.sklearn import DeepSymbolicRegressor

    start = time.time()
    reg = DeepSymbolicRegressor(copy.deepcopy(dso_config))
    reg.fit(x_train, y_train)
    runtime_seconds = time.time() - start

    # Extract best expression
    best_program = reg.program_
    best_expr = repr(best_program.sympy_expr)

    # Evaluate on train/test
    train_pred = best_program.execute(x_train).reshape(-1)
    test_pred = best_program.execute(x_test).reshape(-1)
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    result = {
        "algorithm": "DSR" if str(wrapper_cfg.get("mode", "dsr")).lower() == "dsr" else "uDSR",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": n_variables,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "runtime_seconds": runtime_seconds,
        "best_expression": best_expr,
        "seed": int(dso_config["experiment"]["seed"]),
        "status": "success",
    }

    result = apply_rag_sr_result_schema(result)
    summary_path = os.path.join(func_output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSO (DSR / uDSR) baseline on benchmark datasets."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "dsr", "config.yml"),
        help="Path to DSR wrapper config YAML.",
    )
    parser.add_argument(
        "--function-id", type=int, default=None, help="Run one benchmark function."
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmark functions.")
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper_cfg = load_yaml(resolve_path(args.config))
    metadata = load_metadata(resolve_path(wrapper_cfg["BENCHMARK_METADATA_PATH"]))

    if args.all:
        function_ids = sorted(metadata.keys())
    elif args.function_id is not None:
        if args.function_id not in metadata:
            raise ValueError(f"Unknown function id: {args.function_id}")
        function_ids = [args.function_id]
    else:
        raise ValueError("Please provide --function-id N or use --all.")

    output_root = resolve_path(wrapper_cfg["OUTPUT_PATH"])
    ensure_output_dir(output_root)
    summary_path = os.path.join(output_root, "summary.jsonl")

    from utils.readAndwrite import append_jsonl

    all_results = []
    for function_id in function_ids:
        print(f"[{wrapper_cfg.get('mode', 'DSR')}] Running func{function_id} ({metadata[function_id]['name']})")
        result = train_one(
            function_id=function_id, metadata=metadata, wrapper_cfg=wrapper_cfg
        )
        all_results.append(result)
        print(
            f"[{wrapper_cfg.get('mode', 'DSR')}] func{function_id} done | "
            f"train_mse={result['train_mse']:.6g} | "
            f"test_mse={result['test_mse']:.6g} | "
            f"time={result['runtime_seconds']:.2f}s"
        )

    append_jsonl(summary_path, all_results)
    print(f"[{wrapper_cfg.get('mode', 'DSR')}] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
