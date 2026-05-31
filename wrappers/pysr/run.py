import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sympy
import yaml


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.paths import resolve_deps_path
from utils.readAndwrite import append_jsonl, write_json

PY_SR_SOURCE_DIR = resolve_deps_path("iclrcom", "PySR")
from utils.baseline_result_schema import apply_rag_sr_result_schema


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


def prepare_runtime_env(config):
    runtime_root = resolve_path(config.get("RUNTIME_ROOT", ".pysr_runtime"))
    project_dir = os.path.join(runtime_root, "project")
    depot_dir = os.path.join(runtime_root, "depot")
    ensure_output_dir(project_dir)
    ensure_output_dir(depot_dir)
    julia_exe = os.path.join(project_dir, "pyjuliapkg", "install", "bin", "julia.exe")
    os.environ["PYTHON_JULIAPKG_PROJECT"] = project_dir
    os.environ["JULIA_DEPOT_PATH"] = depot_dir
    if os.path.exists(julia_exe):
        os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
    else:
        os.environ.pop("PYTHON_JULIAPKG_OFFLINE", None)
    return runtime_root, project_dir, depot_dir


def load_xy(dataset_root, function_id, n_variables, meta_entry=None):
    from utils.data_loader import resolve_evosr, load_data as _ld
    fp_e, inp, tgt = resolve_evosr(meta_entry) if meta_entry else (None, None, None)
    if fp_e:
        Xtr, ytr, Xte, yte = _ld(fp_e, inp, tgt)
        return Xtr, ytr, Xte, yte, fp_e["train_data"], fp_e["test_data"]

    train_path = os.path.join(dataset_root, f"fitness_cases{function_id}.csv")
    test_path = os.path.join(dataset_root, f"hold_out{function_id}.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feature_columns = ["x1"] if int(n_variables) == 1 else ["x1", "x2"]
    x_train = df_train[feature_columns].copy()
    y_train = df_train["y"].to_numpy()
    x_test = df_test[feature_columns].copy()
    y_test = df_test["y"].to_numpy()
    return x_train, y_train, x_test, y_test, train_path, test_path


def import_pysr():
    try:
        from pysr import PySRRegressor
    except ModuleNotFoundError as exc:
        if PY_SR_SOURCE_DIR not in sys.path:
            sys.path.insert(0, PY_SR_SOURCE_DIR)
        try:
            from pysr import PySRRegressor
        except ModuleNotFoundError:
            raise RuntimeError(
                "PySR wrapper could not import its runtime dependencies. "
                "The repository source exists locally, but the active environment is still missing "
                "required packages such as `juliacall` or `pysr` runtime dependencies."
            ) from exc
    return PySRRegressor


def build_model(config, output_root):
    PySRRegressor = import_pysr()
    parallelism = str(config.get("parallelism", "serial"))
    model_kwargs = dict(
        model_selection=config.get("model_selection", "best"),
        niterations=int(config.get("niterations", 50)),
        populations=int(config.get("populations", 15)),
        population_size=int(config.get("population_size", 100)),
        maxsize=int(config.get("maxsize", 30)),
        progress=bool(config.get("progress", False)),
        verbosity=int(config.get("verbosity", 0)),
        deterministic=bool(config.get("deterministic", True)),
        random_state=int(config.get("seed", 42)),
        parallelism=parallelism,
        binary_operators=list(config.get("binary_operators", ["+", "-", "*", "/"])),
        unary_operators=list(config.get("unary_operators", ["sin", "cos"])),
        extra_sympy_mappings={
            "square": lambda x: x**2,
            "protect_sqrt": lambda x: sympy.sqrt(sympy.Abs(x)),
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        temp_equation_file=False,
        output_directory=output_root,
    )
    if parallelism != "serial":
        model_kwargs["procs"] = int(config.get("procs", 0))
    return PySRRegressor(**model_kwargs)


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


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
        meta_entry=meta,
    )

    model = build_model(config, raw_dir)

    start = time.time()
    model.fit(x_train, y_train)
    runtime_seconds = time.time() - start

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    equations_df = model.equations_.copy()
    equations_path = os.path.join(func_output_dir, "equations.csv")
    equations_df.to_csv(equations_path, index=False)

    best_row = model.get_best()
    best_expression = str(model.sympy())
    raw_run_directory = str(os.path.join(model.output_directory_, model.run_id_))

    result = {
        "algorithm": "PySR",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "train_path": train_path,
        "test_path": test_path,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "runtime_seconds": runtime_seconds,
        "best_expression": best_expression,
        "best_loss": float(best_row["loss"]),
        "best_complexity": int(best_row["complexity"]),
        "model_selection": config.get("model_selection", "best"),
        "run_directory": raw_run_directory,
        "equations_path": equations_path,
        "status": "success",
    }

    result = apply_rag_sr_result_schema(result)
    write_json(os.path.join(func_output_dir, "run_summary.json"), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PySR baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "pysr", "config.yml"),
        help="Path to PySR wrapper config YAML.",
    )
    parser.add_argument("--function-id", type=int, default=None, help="Run one benchmark function.")
    parser.add_argument("--all", action="store_true", help="Run all benchmark functions.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    _, project_dir, depot_dir = prepare_runtime_env(config)
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
        print(f"[PySR] Running func{function_id} ({metadata[function_id]['name']})")
        print(f"[PySR] runtime_project={project_dir}")
        print(f"[PySR] runtime_depot={depot_dir}")
        result = train_one(function_id=function_id, metadata=metadata, config=config)
        all_results.append(result)
        print(
            f"[PySR] func{function_id} done | "
            f"train_mse={result['train_mse']:.6g} | "
            f"test_mse={result['test_mse']:.6g} | "
            f"time={result['runtime_seconds']:.2f}s"
        )

    append_jsonl(summary_path, all_results)
    print(f"[PySR] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
