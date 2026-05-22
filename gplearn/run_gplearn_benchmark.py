import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.data_loader import load_data
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.readAndwrite import append_jsonl, write_json


EPS = 1e-9


def _protected_division(x1, x2):
    return np.divide(x1, x2, out=np.zeros_like(x1, dtype=float), where=np.abs(x2) > EPS)


def _protected_sqrt(x1):
    return np.sqrt(x1, out=np.zeros_like(x1, dtype=float), where=x1 > EPS)


def _square(x1):
    return np.square(x1)


PROTECT_DIV = make_function(function=_protected_division, name="protect_div", arity=2)
PROTECT_SQRT = make_function(function=_protected_sqrt, name="protect_sqrt", arity=1)
SQUARE = make_function(function=_square, name="square", arity=1)


CUSTOM_FUNCTIONS = {
    "protect_div": PROTECT_DIV,
    "protect_sqrt": PROTECT_SQRT,
    "square": SQUARE,
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {int(item["function_id"]): item for item in items}


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(ROOT_DIR, path_value))


def build_file_paths(dataset_dir, function_id):
    return {
        "train_data": os.path.join(dataset_dir, f"fitness_cases{function_id}.csv"),
        "test_data": os.path.join(dataset_dir, f"hold_out{function_id}.csv"),
    }


def get_function_set(config):
    raw_functions = config.get("function_set", [])
    resolved = []
    for name in raw_functions:
        resolved.append(CUSTOM_FUNCTIONS.get(name, name))
    return resolved


def select_features(X, n_variables):
    if int(n_variables) == 1:
        return X[:, [0]]
    return X[:, : int(n_variables)]


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def train_one(function_id, metadata, config):
    dataset_dir = resolve_path(config["DATASET_PATH"])
    output_root = resolve_path(config["OUTPUT_PATH"])
    file_paths = build_file_paths(dataset_dir, function_id)
    X_train, y_train, X_test, y_test = load_data(file_paths)

    n_variables = int(metadata[function_id]["n_variables"])
    X_train = select_features(X_train, n_variables)
    X_test = select_features(X_test, n_variables)

    estimator = SymbolicRegressor(
        population_size=int(config["population_size"]),
        generations=int(config["generations"]),
        tournament_size=int(config["tournament_size"]),
        stopping_criteria=float(config["stopping_criteria"]),
        const_range=tuple(config["const_range"]),
        init_depth=tuple(config["init_depth"]),
        init_method=config["init_method"],
        function_set=tuple(get_function_set(config)),
        metric=config.get("metric", "mse"),
        parsimony_coefficient=float(config["parsimony_coefficient"]),
        p_crossover=float(config["p_crossover"]),
        p_subtree_mutation=float(config["p_subtree_mutation"]),
        p_hoist_mutation=float(config["p_hoist_mutation"]),
        p_point_mutation=float(config["p_point_mutation"]),
        max_samples=float(config["max_samples"]),
        verbose=int(config.get("verbose", 1)),
        random_state=int(config["seed"]),
    )

    start_time = time.time()
    estimator.fit(X_train, y_train)
    elapsed_s = time.time() - start_time

    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    train_mse = float(np.mean((train_pred - y_train) ** 2))
    test_mse = float(np.mean((test_pred - y_test) ** 2))
    expression = str(estimator._program)

    run_result = {
        "algorithm": "gplearn",
        "function_id": function_id,
        "function_name": metadata[function_id]["name"],
        "benchmark_name": metadata[function_id]["name"],
        "n_variables": n_variables,
        "best_expression": expression,
        "expression": expression,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "runtime_seconds": elapsed_s,
        "seed": int(config["seed"]),
        "population_size": int(config["population_size"]),
        "generations": int(config["generations"]),
        "status": "success",
    }
    run_result = apply_rag_sr_result_schema(
        run_result,
        best_generation=int(config["generations"]),
        final_generation=int(config["generations"]),
        n_logged_generations=int(config["generations"]) + 1,
    )

    func_output_dir = os.path.join(output_root, f"func{function_id}")
    ensure_output_dir(func_output_dir)
    write_json(os.path.join(func_output_dir, "run_summary.json"), run_result)

    history_entries = []
    run_details = estimator.run_details_
    total_rows = len(run_details["generation"])
    for idx in range(total_rows):
        history_entries.append(
            {
                "function_id": function_id,
                "generation": int(run_details["generation"][idx]),
                "average_length": float(run_details["average_length"][idx]),
                "average_fitness": float(run_details["average_fitness"][idx]),
                "best_length": float(run_details["best_length"][idx]),
                "best_fitness": float(run_details["best_fitness"][idx]),
                "generation_time": float(run_details["generation_time"][idx]),
            }
        )
    append_jsonl(os.path.join(func_output_dir, "generation_history.jsonl"), history_entries)

    return run_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run gplearn baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("baselines", "gplearn", "config_gplearn.yml"),
        help="Path to gplearn YAML config.",
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
    for function_id in function_ids:
        print(f"[gplearn] Running func{function_id} ({metadata[function_id]['name']})")
        result = train_one(function_id=function_id, metadata=metadata, config=config)
        all_results.append(result)
        print(
            f"[gplearn] func{function_id} done | "
            f"train_mse={result['train_mse']:.6g} | "
            f"test_mse={result['test_mse']:.6g} | "
            f"time={result['runtime_seconds']:.2f}s"
        )

    append_jsonl(summary_path, all_results)
    print(f"[gplearn] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
