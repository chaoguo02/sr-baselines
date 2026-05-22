import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.readAndwrite import append_jsonl, write_json
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path

UPSTREAM_ROOT = resolve_deps_path("external", "GP-GOMEA")
UPSTREAM_PYTHONPKG = os.path.join(UPSTREAM_ROOT, "pythonpkg")


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

    if int(n_variables) == 1:
        feature_columns = ["x1"]
    elif int(n_variables) == 2:
        feature_columns = ["x1", "x2"]
    else:
        raise ValueError(
            f"GP-GOMEA wrapper currently supports 1D/2D benchmark tasks, got n_variables={n_variables}."
        )

    x_train = df_train[feature_columns].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    x_test = df_test[feature_columns].to_numpy(dtype=float)
    y_test = df_test["y"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, train_path, test_path


def import_gpgomea_regressor():
    if UPSTREAM_PYTHONPKG not in sys.path:
        sys.path.insert(0, UPSTREAM_PYTHONPKG)

    try:
        from pyGPGOMEA import GPGOMEARegressor
    except Exception as exc:
        raise RuntimeError(
            "GP-GOMEA wrapper found the upstream source tree, but could not import "
            "its native Python extension. The upstream project requires a compiled "
            "module exposed through `pyGPGOMEA`. On this machine, that usually means "
            "the C++ toolchain build has not been completed yet."
        ) from exc
    return GPGOMEARegressor


def build_model(config):
    GPGOMEARegressor = import_gpgomea_regressor()
    return GPGOMEARegressor(
        time=int(config.get("time", 30)),
        generations=int(config.get("generations", -1)),
        evaluations=int(config.get("evaluations", -1)),
        linearscaling=bool(config.get("linearscaling", True)),
        functions=str(config.get("functions", "+_-_*_p/_sqrt_plog")),
        erc=bool(config.get("erc", True)),
        gomea=bool(config.get("gomea", True)),
        gomfos=str(config.get("gomfos", "LT")),
        ims=str(config.get("ims", "4_1")),
        popsize=int(config.get("popsize", 64)),
        batchsize=int(config.get("batchsize", 256)),
        initmaxtreeheight=int(config.get("initmaxtreeheight", 3)),
        maxtreeheight=int(config.get("maxtreeheight", 17)),
        maxsize=int(config.get("maxsize", 1000)),
        seed=int(config.get("seed", 42)),
        parallel=int(config.get("parallel", 1)),
        silent=bool(config.get("silent", True)),
    )


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def normalize_expression(expression):
    text = str(expression)
    return text.replace("p/", "/").replace("plog", "log")


def train_one(function_id, metadata, config):
    dataset_root = resolve_path(config["DATASET_PATH"])
    output_root = resolve_path(config["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    raw_dir = os.path.join(func_output_dir, "raw")
    ensure_output_dir(raw_dir)

    meta = metadata[function_id]
    x_train, y_train, x_test, y_test, train_path, test_path = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=meta["n_variables"],
    )

    model = build_model(config)

    start = time.time()
    model.fit(x_train, y_train)
    runtime_seconds = time.time() - start

    train_pred = np.asarray(model.predict(x_train), dtype=float).reshape(-1)
    test_pred = np.asarray(model.predict(x_test), dtype=float).reshape(-1)
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    best_expression = normalize_expression(model.get_model())
    evaluations = int(model.get_evaluations())
    n_nodes = int(model.get_n_nodes())

    progress_log_path = os.path.join(func_output_dir, "progress_log.json")
    try:
        write_json(progress_log_path, model.get_progress_log())
    except Exception:
        progress_log_path = None

    result = {
        "algorithm": "GP-GOMEA",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "train_path": train_path,
        "test_path": test_path,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "runtime_seconds": runtime_seconds,
        "best_expression": best_expression,
        "evaluations": evaluations,
        "n_nodes": n_nodes,
        "search_algorithm": "gomea" if bool(config.get("gomea", True)) else "gp",
        "status": "success",
    }
    if progress_log_path is not None:
        result["progress_log_path"] = progress_log_path

    result = apply_rag_sr_result_schema(result)
    write_json(os.path.join(func_output_dir, "run_summary.json"), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GP-GOMEA baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "gp_gomea", "config.yml"),
        help="Path to GP-GOMEA wrapper config YAML.",
    )
    parser.add_argument("--function-id", type=int, default=None, help="Run one benchmark function.")
    parser.add_argument("--all", action="store_true", help="Run all benchmark functions.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    metadata = load_metadata(resolve_path(config["BENCHMARK_METADATA_PATH"]))

    if args.all:
        function_ids = sorted(metadata.keys())
    elif args.function_id is not None:
        if args.function_id not in metadata:
            raise ValueError(f"Unknown function id: {args.function_id}")
        function_ids = [args.function_id]
    else:
        raise ValueError("Please provide --function-id N or use --all.")

    output_root = resolve_path(config["OUTPUT_PATH"])
    ensure_output_dir(output_root)
    summary_path = os.path.join(output_root, "summary.jsonl")

    all_results = []
    try:
        for function_id in function_ids:
            print(f"[GP-GOMEA] Running func{function_id} ({metadata[function_id]['name']})")
            result = train_one(function_id=function_id, metadata=metadata, config=config)
            all_results.append(result)
            print(
                f"[GP-GOMEA] func{function_id} done | "
                f"train_mse={result['train_mse']:.6g} | "
                f"test_mse={result['test_mse']:.6g} | "
                f"time={result['runtime_seconds']:.2f}s"
            )
    except RuntimeError as exc:
        raise SystemExit(f"[GP-GOMEA] {exc}") from exc

    append_jsonl(summary_path, all_results)
    print(f"[GP-GOMEA] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
