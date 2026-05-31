import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import yaml


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JULIA_RUNNER = os.path.join(ROOT_DIR, "wrappers", "lasr", "run_lasr.jl")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.env_loader import load_env_file, resolve_api_base_url, resolve_api_key
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path
from utils.readAndwrite import append_jsonl, read_json, write_json


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


def get_julia_bin(config):
    runtime_root = resolve_path(config.get("RUNTIME_ROOT", ".pysr_runtime"))
    julia_bin = os.path.join(runtime_root, "project", "pyjuliapkg", "install", "bin", "julia.exe")
    if not os.path.exists(julia_bin):
        raise RuntimeError(
            "LaSR wrapper could not find a Julia runtime. "
            "Expected an existing Julia binary at "
            f"`{julia_bin}`."
        )
    return julia_bin


def get_julia_runtime_paths(config):
    runtime_root = resolve_path(config.get("RUNTIME_ROOT", ".pysr_runtime"))
    project_dir = os.path.join(runtime_root, "project")
    depot_dir = os.path.join(runtime_root, "depot")
    return runtime_root, project_dir, depot_dir


def get_api_key(config):
    env_name = str(config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = resolve_api_key(env_name, "OPENAI_API_KEY", "API_KEY", "DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LaSR wrapper requires an API key because `use_llm` is enabled. "
            f"Checked `{env_name}`, `OPENAI_API_KEY`, `API_KEY`, and `DASHSCOPE_API_KEY`."
        )
    return api_key


def load_xy(dataset_root, function_id, n_variables, meta_entry=None):
    from utils.data_loader import resolve_evosr, load_data as _ld
    fp_e, inp, tgt = resolve_evosr(meta_entry) if meta_entry else (None, None, None)
    if fp_e:
        Xtr, ytr, Xte, yte = _ld(fp_e, inp, tgt)
        # Build DataFrames with consistent column naming
        nv = Xtr.shape[1]
        cols = [f"x{i+1}" for i in range(nv)]
        df_tr = pd.DataFrame(Xtr, columns=cols)
        df_tr["y"] = ytr
        df_te = pd.DataFrame(Xte, columns=cols)
        df_te["y"] = yte
        return df_tr, df_te, fp_e["train_data"], fp_e["test_data"], cols

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
            f"LaSR wrapper currently supports 1D/2D benchmark tasks, got n_variables={n_variables}."
        )

    return (
        df_train[feature_columns + ["y"]].copy(),
        df_test[feature_columns + ["y"]].copy(),
        train_path,
        test_path,
        feature_columns,
    )


def write_matrix_csv(df, path):
    ensure_output_dir(os.path.dirname(path))
    df.to_csv(path, index=False, header=False)


def build_context(config, feature_columns):
    variables = ", ".join(feature_columns)
    template = str(
        config.get(
            "llm_context_template",
            "We are searching for a symbolic regression equation over variables {variables}.",
        )
    )
    return template.format(variables=variables)


def decode_process_output(payload):
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8", errors="replace")


def build_runner_config(function_id, meta, config, feature_columns, train_csv, test_csv, func_output_dir):
    return {
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "seed": int(config.get("seed", 42)),
        "train_csv": train_csv,
        "test_csv": test_csv,
        "output_dir": func_output_dir,
        "project_dir": resolve_deps_path("external", "LibraryAugmentedSymbolicRegression.jl"),
        "prompts_dir": resolve_deps_path("external", "LibraryAugmentedSymbolicRegression.jl", "prompts") + os.sep,
        "api_key_env": str(config.get("api_key_env", "OPENAI_API_KEY")),
        "api_url": resolve_api_base_url(
            config.get("api_url"),
            "https://api.openai.com/v1",
            "OPENAI_BASE_URL",
            "QWEN_BASE_URL",
        ),
        "api_max_tokens": int(config.get("api_max_tokens", 1000)),
        "model_name": str(config.get("model_name", "gpt-4o-mini")),
        "llm_context": build_context(config, feature_columns),
        "binary_operators": list(config.get("binary_operators", ["+", "-", "*", "/", "^"])),
        "unary_operators": list(config.get("unary_operators", ["sin", "cos", "exp", "log", "sqrt"])),
        "niterations": int(config.get("niterations", 10)),
        "populations": int(config.get("populations", 10)),
        "parallelism": str(config.get("parallelism", "serial")),
        "use_llm": bool(config.get("use_llm", True)),
        "use_concepts": bool(config.get("use_concepts", True)),
        "use_concept_evolution": bool(config.get("use_concept_evolution", True)),
        "llm_probability": float(config.get("llm_probability", 1.0e-4)),
        "num_generated_equations": int(config.get("num_generated_equations", 5)),
        "num_generated_concepts": int(config.get("num_generated_concepts", 5)),
        "num_pareto_context": int(config.get("num_pareto_context", 5)),
        "max_concepts": int(config.get("max_concepts", 30)),
        "is_parametric": bool(config.get("is_parametric", False)),
        "verbose": bool(config.get("verbose", False)),
        "variable_names": {name: name for name in feature_columns},
    }


def train_one(function_id, metadata, config):
    load_env_file()
    dataset_root = resolve_path(config["DATASET_PATH"])
    output_root = resolve_path(config["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    raw_input_dir = os.path.join(func_output_dir, "raw_inputs")
    ensure_output_dir(raw_input_dir)

    meta = metadata[function_id]
    train_df, test_df, train_path, test_path, feature_columns = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=meta["n_variables"],
        meta_entry=meta,
    )
    train_csv = os.path.join(raw_input_dir, "train.csv")
    test_csv = os.path.join(raw_input_dir, "test.csv")
    write_matrix_csv(train_df, train_csv)
    write_matrix_csv(test_df, test_csv)

    runner_cfg = build_runner_config(
        function_id=function_id,
        meta=meta,
        config=config,
        feature_columns=feature_columns,
        train_csv=train_csv,
        test_csv=test_csv,
        func_output_dir=func_output_dir,
    )
    runner_cfg_path = os.path.join(func_output_dir, "runner_config.json")
    write_json(runner_cfg_path, runner_cfg)

    julia_bin = get_julia_bin(config)
    _, project_dir, depot_dir = get_julia_runtime_paths(config)
    if bool(config.get("use_llm", True)):
        get_api_key(config)
    cmd = [julia_bin, "--project=" + runner_cfg["project_dir"], JULIA_RUNNER, runner_cfg_path]
    env = os.environ.copy()
    env["JULIA_DEPOT_PATH"] = depot_dir
    env["PYTHON_JULIAPKG_PROJECT"] = project_dir
    completed = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, env=env)

    stdout_path = os.path.join(func_output_dir, "julia_stdout.log")
    stderr_path = os.path.join(func_output_dir, "julia_stderr.log")
    stdout_text = decode_process_output(completed.stdout)
    stderr_text = decode_process_output(completed.stderr)
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(stdout_text)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(stderr_text)

    if completed.returncode != 0:
        message = (
            "LaSR Julia runner failed. "
            f"See `{stdout_path}` and `{stderr_path}` for details."
        )
        if "Pkg.instantiate()" in stderr_text or "does not seem to be installed" in stderr_text:
            message = (
                "LaSR Julia runner failed because the upstream Julia environment has not been instantiated yet. "
                f"Please run `Pkg.instantiate()` for `LibraryAugmentedSymbolicRegression.jl`."
                f"See `{stdout_path}` and `{stderr_path}` for details."
            )
        raise RuntimeError(
            message
        )

    result_path = os.path.join(func_output_dir, "run_summary.json")
    result = read_json(result_path)
    result["train_path"] = train_path
    result["test_path"] = test_path
    result["runner_config_path"] = runner_cfg_path
    result["julia_stdout_log"] = stdout_path
    result["julia_stderr_log"] = stderr_path
    result = apply_rag_sr_result_schema(result)
    write_json(result_path, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LaSR baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "lasr", "config.yml"),
        help="Path to LaSR wrapper config YAML.",
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
            print(f"[LaSR] Running func{function_id} ({metadata[function_id]['name']})")
            result = train_one(function_id=function_id, metadata=metadata, config=config)
            all_results.append(result)
            print(
                f"[LaSR] func{function_id} done | "
                f"train_mse={result['train_mse']:.6g} | "
                f"test_mse={result['test_mse']:.6g} | "
                f"time={result['runtime_seconds']:.2f}s"
            )
    except RuntimeError as exc:
        raise SystemExit(f"[LaSR] {exc}") from exc

    append_jsonl(summary_path, all_results)
    print(f"[LaSR] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
