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

from utils.env_loader import load_env_file, resolve_api_key, resolve_api_base_url
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path
from utils.readAndwrite import append_jsonl, write_json

UPSTREAM_ROOT = resolve_deps_path("external", "SR-LLM")


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
        variable_names = ["x1"]
    elif int(n_variables) == 2:
        feature_columns = ["x1", "x2"]
        variable_names = ["x1", "x2"]
    else:
        raise ValueError(
            f"SR-LLM wrapper currently supports 1D/2D benchmark tasks, got n_variables={n_variables}."
        )

    x_train = df_train[feature_columns].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    x_test = df_test[feature_columns].to_numpy(dtype=float)
    y_test = df_test["y"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, train_path, test_path, variable_names


def _add_langchain_compat():
    """Add compatibility shims so upstream SR-LLM can import old langchain APIs."""
    import types
    import langchain_community.callbacks
    _lc_callbacks = types.ModuleType("langchain.callbacks")
    _lc_callbacks.OpenAICallbackHandler = langchain_community.callbacks.OpenAICallbackHandler
    _lc_callbacks.__package__ = "langchain.callbacks"
    sys.modules["langchain.callbacks"] = _lc_callbacks

    # Shim langchain.schema (messages moved to langchain_core.messages in v1.x)
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    _lc_schema = types.ModuleType("langchain.schema")
    _lc_schema.AIMessage = AIMessage
    _lc_schema.HumanMessage = HumanMessage
    _lc_schema.SystemMessage = SystemMessage
    _lc_schema.__package__ = "langchain.schema"
    sys.modules["langchain.schema"] = _lc_schema

    # Also shim ChatOpenAI which moved to langchain_community in v1.x
    try:
        from langchain_community.chat_models import ChatOpenAI as _ChatOpenAI
    except ImportError:
        _ChatOpenAI = None
    if _ChatOpenAI:
        _lc_chat = types.ModuleType("langchain.chat_models")
        _lc_chat.ChatOpenAI = _ChatOpenAI
        _lc_chat.__package__ = "langchain.chat_models"
        sys.modules["langchain.chat_models"] = _lc_chat


def import_sr_llm(model_name="gpt-4o-mini"):
    try:
        if UPSTREAM_ROOT not in sys.path:
            sys.path.insert(0, UPSTREAM_ROOT)
        _add_langchain_compat()
        # Set env vars for OpenAI-compatible API before upstream imports ChatOpenAI
        _api_key = resolve_api_key("OPENAI_API_KEY", "API_KEY", "DASHSCOPE_API_KEY")
        if _api_key:
            os.environ["OPENAI_API_KEY"] = _api_key
        _base_url = resolve_api_base_url(
            None, "https://api.openai.com/v1",
            "OPENAI_BASE_URL", "QWEN_BASE_URL",
        )
        if _base_url:
            os.environ["OPENAI_API_BASE"] = _base_url
            os.environ["OPENAI_BASE_URL"] = _base_url

        import torch
        from codes.applications.general_symbolic_regression import general_symbolic_regression

        # Monkey-patch Agent.build_path_var to not overwrite our API key
        import codes.trafficSR.D_updation_by_LLM.Agent as _agent_mod
        _orig_build = _agent_mod.Agent.build_path_var
        _orig_init = _agent_mod.Agent.__init__
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Override model type assertion; the actual model is determined by api_base
            self.model_type = model_name  # from wrapper_cfg, default qwen3.5-plus
            # Override the hardcoded base_url in the ChatOpenAI client
            # (upstream code uses a hardcoded proxy endpoint, but we need the Qwen-compatible API)
            if _base_url and hasattr(self, 'llm'):
                base = _base_url.rstrip('/')
                if not base.endswith('/v1'):
                    base = base + '/v1'
                self.llm.openai_api_base = base
        _agent_mod.Agent.__init__ = _patched_init
        def _patched_build_path_var(self):
            _orig_build(self)
            if _api_key:
                os.environ["OPENAI_API_KEY"] = _api_key
            if _base_url:
                os.environ["OPENAI_API_BASE"] = _base_url
                os.environ["OPENAI_BASE_URL"] = _base_url
        _agent_mod.Agent.build_path_var = _patched_build_path_var
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            "SR-LLM wrapper found the upstream source tree, but the active environment is still "
            f"missing the runtime dependency `{missing_name}`. "
            "The active environment likely still needs packages from "
            "`baselines/external/SR-LLM/environment.yml` (see --deps-dir)."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "SR-LLM wrapper found the upstream source tree, but could not import its runtime "
            "dependencies. The active environment likely still needs packages from "
            "`baselines/external/SR-LLM/environment.yml`, plus torch/torchvision."
        ) from exc
    return general_symbolic_regression, torch


def snapshot_run_dirs():
    snapshots = {}
    for folder_name in ["trainResult", "physo_log"]:
        folder = os.path.join(UPSTREAM_ROOT, folder_name)
        ensure_output_dir(folder)
        snapshots[folder_name] = set(os.listdir(folder))
    return snapshots


def detect_new_run_dirs(before_snapshot):
    raw = {}
    for folder_name in ["trainResult", "physo_log"]:
        folder = os.path.join(UPSTREAM_ROOT, folder_name)
        current = set(os.listdir(folder))
        created = sorted(current - before_snapshot.get(folder_name, set()))
        raw[folder_name] = [os.path.join(folder, name) for name in created]
    return raw


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def evaluate_program(best_function, x):
    import torch

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_pred = best_function(x_tensor)
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()
    return np.asarray(y_pred, dtype=float).reshape(-1)


def build_variable_descriptions(variable_names, function_name):
    descriptions = []
    for idx, name in enumerate(variable_names, start=1):
        descriptions.append(f"input variable {idx} ({name}) for benchmark function {function_name}")
    return descriptions


def parse_operator_tokens(wrapper_cfg):
    return list(
        wrapper_cfg.get(
            "operator_tokens",
            ["add", "mul", "sub", "div", "n2", "sqrt", "sin", "cos", "exp", "log", "pow"],
        )
    )


def parse_free_const_setup(wrapper_cfg):
    tokens = list(wrapper_cfg.get("free_const_tokens", ["c_1", "c_2", "c_3"]))
    units = list(wrapper_cfg.get("free_const_units", [[0, 0] for _ in tokens]))
    descriptions = list(
        wrapper_cfg.get(
            "free_const_descriptions",
            [f"free constant {idx + 1}" for idx in range(len(tokens))],
        )
    )
    bounds = list(wrapper_cfg.get("free_const_bounds", [[-10.0, 10.0] for _ in tokens]))
    return tokens, units, descriptions, bounds


def parse_fixed_const_setup(wrapper_cfg):
    tokens = list(wrapper_cfg.get("fixed_const_tokens", ["1"]))
    values = list(wrapper_cfg.get("fixed_const_values", [1.0 for _ in tokens]))
    units = list(wrapper_cfg.get("fixed_const_units", [[0, 0] for _ in tokens]))
    descriptions = list(
        wrapper_cfg.get(
            "fixed_const_descriptions",
            [f"fixed constant {idx + 1}" for idx in range(len(tokens))],
        )
    )
    return tokens, values, units, descriptions


def train_one(function_id, metadata, wrapper_cfg):
    load_env_file()
    use_rag = bool(wrapper_cfg.get("use_rag", True))
    if use_rag and not resolve_api_key("OPENAI_API_KEY", "API_KEY", "DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "SR-LLM is configured with `use_rag: true`, but no API key was found. "
            "Checked `OPENAI_API_KEY`, `API_KEY`, and `DASHSCOPE_API_KEY`."
        )

    model_name = str(wrapper_cfg.get("model_name", "qwen3.5-plus"))
    general_symbolic_regression, torch = import_sr_llm(model_name=model_name)
    dataset_root = resolve_path(wrapper_cfg["DATASET_PATH"])
    output_root = resolve_path(wrapper_cfg["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    ensure_output_dir(func_output_dir)

    meta = metadata[function_id]
    x_train, y_train, x_test, y_test, train_path, test_path, variable_names = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=meta["n_variables"],
    )

    variable_units = [[0, 0] for _ in variable_names]
    variable_descriptions = build_variable_descriptions(variable_names, meta["name"])
    free_const_tokens, free_const_units, free_const_descriptions, free_const_bounds = parse_free_const_setup(
        wrapper_cfg
    )
    fixed_const_tokens, fixed_const_values, fixed_const_units, fixed_const_descriptions = parse_fixed_const_setup(
        wrapper_cfg
    )
    memory_path = resolve_deps_path("external", "SR-LLM", "codes/ragLibrary/memory_general")

    input_manifest = {
        "variable_names": variable_names,
        "variable_units": variable_units,
        "variable_descriptions": variable_descriptions,
        "target_name": str(wrapper_cfg.get("target_name", "y")),
        "target_unit": list(wrapper_cfg.get("target_unit", [0, 0])),
        "target_description": str(wrapper_cfg.get("target_description", "benchmark target variable")),
        "operator_tokens": parse_operator_tokens(wrapper_cfg),
        "free_const_tokens": free_const_tokens,
        "free_const_units": free_const_units,
        "free_const_descriptions": free_const_descriptions,
        "free_const_bounds": free_const_bounds,
        "fixed_const_tokens": fixed_const_tokens,
        "fixed_const_values": fixed_const_values,
        "fixed_const_units": fixed_const_units,
        "fixed_const_descriptions": fixed_const_descriptions,
        "use_rag": use_rag,
        "memory_path": memory_path,
        "seed": int(wrapper_cfg.get("seed", 100)),
        "n_epochs": int(wrapper_cfg.get("n_epochs", 30)),
        "n_evolutions": int(wrapper_cfg.get("n_evolutions", 5)),
        "device": str(wrapper_cfg.get("device", "cpu")),
    }
    write_json(os.path.join(func_output_dir, "generated_inputs.json"), input_manifest)

    before_snapshot = snapshot_run_dirs()
    start = time.time()
    best_expression, best_function = general_symbolic_regression(
        x_train,
        y_train,
        variable_names=variable_names,
        variable_units=variable_units,
        variable_descriptions=variable_descriptions,
        target_name=input_manifest["target_name"],
        target_unit=input_manifest["target_unit"],
        target_description=input_manifest["target_description"],
        operator_tokens=input_manifest["operator_tokens"],
        free_const_tokens=free_const_tokens,
        free_const_units=free_const_units,
        free_const_descriptions=free_const_descriptions,
        free_const_bounds=free_const_bounds,
        fixed_const_tokens=fixed_const_tokens,
        fixed_const_values=fixed_const_values,
        fixed_const_units=fixed_const_units,
        fixed_const_descriptions=fixed_const_descriptions,
        seed=input_manifest["seed"],
        n_epochs=input_manifest["n_epochs"],
        n_evolutions=input_manifest["n_evolutions"],
        use_rag=use_rag,
        memory_path=memory_path,
        device=input_manifest["device"],
        address=str(wrapper_cfg.get("address", "172.22.0.1")),
        port=str(wrapper_cfg.get("port", "7890")),
    )
    runtime_seconds = time.time() - start
    new_run_dirs = detect_new_run_dirs(before_snapshot)

    if best_expression is None or best_function is None:
        raise RuntimeError(
            "SR-LLM finished without returning a valid expression. "
            "Please inspect the upstream train logs under the SR-LLM deps directory."
        )

    train_pred = evaluate_program(best_function, x_train)
    test_pred = evaluate_program(best_function, x_test)

    result = {
        "algorithm": "SR-LLM",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "train_path": train_path,
        "test_path": test_path,
        "train_mse": mse(y_train, train_pred),
        "test_mse": mse(y_test, test_pred),
        "runtime_seconds": runtime_seconds,
        "best_expression": str(best_expression),
        "use_rag": use_rag,
        "memory_path": memory_path,
        "n_epochs": input_manifest["n_epochs"],
        "n_evolutions": input_manifest["n_evolutions"],
        "device": input_manifest["device"],
        "upstream_train_result_dirs": new_run_dirs.get("trainResult", []),
        "upstream_physo_log_dirs": new_run_dirs.get("physo_log", []),
        "generated_inputs_path": os.path.join(func_output_dir, "generated_inputs.json"),
        "status": "success",
    }

    result = apply_rag_sr_result_schema(result)
    write_json(os.path.join(func_output_dir, "run_summary.json"), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SR-LLM baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "sr_llm", "config.yml"),
        help="Path to SR-LLM wrapper config YAML.",
    )
    parser.add_argument("--function-id", type=int, default=None, help="Run one benchmark function.")
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

    all_results = []
    try:
        for function_id in function_ids:
            print(f"[SR-LLM] Running func{function_id} ({metadata[function_id]['name']})")
            result = train_one(function_id=function_id, metadata=metadata, wrapper_cfg=wrapper_cfg)
            all_results.append(result)
            print(
                f"[SR-LLM] func{function_id} done | "
                f"train_mse={result['train_mse']:.6g} | "
                f"test_mse={result['test_mse']:.6g} | "
                f"time={result['runtime_seconds']:.2f}s"
            )
    except RuntimeError as exc:
        raise SystemExit(f"[SR-LLM] {exc}") from exc

    append_jsonl(summary_path, all_results)
    print(f"[SR-LLM] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
