import argparse
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.env_loader import load_env_file, resolve_api_base_url, resolve_api_key
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path
from utils.readAndwrite import append_jsonl, write_json

UPSTREAM_ROOT = resolve_deps_path("external", "In-Context-Symbolic-Regression")
UPSTREAM_MAIN_PATH = os.path.join(UPSTREAM_ROOT, "main.py")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def to_repo_relative(path):
    return os.path.relpath(path, ROOT_DIR)


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
            f"ICSR wrapper currently supports 1D/2D benchmark tasks, got n_variables={n_variables}."
        )

    x_train = df_train[feature_columns].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    x_test = df_test[feature_columns].to_numpy(dtype=float)
    y_test = df_test["y"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, train_path, test_path


def _stub_models_module():
    """Patch `models` so upstream utils.py can import without transformers/LLaVA.

    The upstream models/__init__.py normally imports HuggingFaceModel (needs
    transformers), LLaVaModelHF (needs transformers+LLaVA), and OpenAIModel
    (pure Python).  We replace __init__.py with a version that imports only
    OpenAIModel (from the real openai_model.py) and stubs the other two.
    """
    import types as _types
    import importlib.util as _import_util

    # Load the real openai_model.py (pure Python, no heavy deps)
    om_path = os.path.join(UPSTREAM_ROOT, "models", "openai_model.py")
    om_spec = _import_util.spec_from_file_location("models.openai_model", om_path)
    if om_spec and om_spec.loader:
        om_mod = _import_util.module_from_spec(om_spec)
        sys.modules["models.openai_model"] = om_mod
        om_spec.loader.exec_module(om_mod)
    else:
        om_mod = _types.ModuleType("models.openai_model")

    # Stub submodules that require transformers
    for name in ("hf_model", "llava_model_hf"):
        sub = _types.ModuleType(f"models.{name}")
        sub.__file__ = os.path.join(UPSTREAM_ROOT, "models", f"{name}.py")
        sys.modules[f"models.{name}"] = sub

    # Build a custom models __init__ that provides all three names
    class _StubHFModel:
        def __init__(self, *args, **kwargs):
            pass

    mod = _types.ModuleType("models")
    mod.HuggingFaceModel = _StubHFModel
    mod.LLaVaModelHF = _StubHFModel
    mod.OpenAIModel = om_mod.OpenAIModel if hasattr(om_mod, "OpenAIModel") else _StubHFModel
    mod.__all__ = ["HuggingFaceModel", "LLaVaModelHF", "OpenAIModel"]
    mod.__file__ = os.path.join(UPSTREAM_ROOT, "models", "__init__.py")
    sys.modules["models"] = mod


def import_icsr_modules():
    try:
        if "fcntl" not in sys.modules:
            sys.modules["fcntl"] = types.ModuleType("fcntl")
        _stub_models_module()
        if UPSTREAM_ROOT not in sys.path:
            sys.path.insert(0, UPSTREAM_ROOT)
        utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(UPSTREAM_ROOT, "utils.py"))
        if utils_spec is None or utils_spec.loader is None:
            raise RuntimeError("Could not create import spec for ICSR utils.py.")
        upstream_utils = importlib.util.module_from_spec(utils_spec)
        sys.modules["utils"] = upstream_utils
        utils_spec.loader.exec_module(upstream_utils)
        spec = importlib.util.spec_from_file_location("icsr_main", UPSTREAM_MAIN_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not create import spec for ICSR main.py.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        icsr_utils = upstream_utils
    except Exception as exc:
        raise RuntimeError(
            "ICSR wrapper found the upstream source tree, but could not import its runtime "
            "dependencies. The active environment likely still needs packages from "
            "`baselines/external/In-Context-Symbolic-Regression/requirements.txt` (see --deps-dir)."
        ) from exc
    return module.Workspace, icsr_utils


def prepare_api_environment(wrapper_cfg):
    load_env_file()
    api_key = resolve_api_key("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "API_KEY")
    api_base_url = resolve_api_base_url(
        wrapper_cfg.get("api_base_url"),
        "https://api.openai.com/v1",
        "OPENAI_BASE_URL",
        "QWEN_BASE_URL",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["ICSR_OPENAI_BASE_URL"] = api_base_url
    return api_key, os.environ["ICSR_OPENAI_BASE_URL"]


def patch_openai_model():
    import openai
    from models import openai_model

    original_init = openai_model.OpenAIModel.__init__
    if getattr(original_init, "_icsr_wrapper_patched", False):
        return

    def wrapped_init(self, model_name, device, dtype, cache_dir=None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.api_key_path = None if "api_key_path" not in kwargs else kwargs["api_key_path"]
        self.api_key = self.get_api_key()
        self.organization_id_path = None if "organization_id_path" not in kwargs else kwargs["organization_id_path"]
        self.organization_id = self.get_org_id()
        assert self.api_key is not None, "API key not found."

        client_kwargs = {
            "api_key": self.api_key,
            "organization": self.organization_id,
        }
        base_url = os.environ.get("ICSR_OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = openai.Client(**client_kwargs)

        self.top_p = 0.9 if "top_p" not in kwargs else kwargs["top_p"]
        self.temperature = 1.0 if "temperature" not in kwargs else kwargs["temperature"]
        self.max_length = 1024 if "max_length" not in kwargs else kwargs["max_length"]
        self.num_return_sequences = 1 if "num_return_sequences" not in kwargs else kwargs["num_return_sequences"]
        self.seed = None if "seed" not in kwargs else kwargs["seed"]

    wrapped_init._icsr_wrapper_patched = True
    openai_model.OpenAIModel.__init__ = wrapped_init


def patch_utils_load_model(icsr_utils, wrapper_cfg):
    original_load_model = icsr_utils.load_model
    if getattr(original_load_model, "_icsr_wrapper_patched", False):
        return

    force_openai_model = bool(wrapper_cfg.get("force_openai_model", False))

    def wrapped_load_model(model_name, device, dtype, cache_dir=None, model_args=None):
        model_args = model_args or {}
        if force_openai_model:
            from models import OpenAIModel

            return OpenAIModel(model_name, device, dtype, cache_dir, **model_args)
        return original_load_model(model_name, device, dtype, cache_dir, model_args)

    wrapped_load_model._icsr_wrapper_patched = True
    icsr_utils.load_model = wrapped_load_model


def build_cfg(wrapper_cfg, function_id, meta, data_folder_rel):
    return OmegaConf.create(
        {
            "output_dir": wrapper_cfg.get("OUTPUT_PATH", "baselines/results/icsr"),
            "max_retries": int(wrapper_cfg.get("max_retries", 5)),
            "force_valid": bool(wrapper_cfg.get("force_valid", False)),
            "force_unique": bool(wrapper_cfg.get("force_unique", False)),
            "prompts_path": resolve_deps_path("external", "In-Context-Symbolic-Regression", "prompts"),
            "max_points_in_prompt": int(wrapper_cfg.get("max_points_in_prompt", 40)),
            "checkpoints": list(wrapper_cfg.get("checkpoints", [])),
            "device": "cpu",
            "use_bfloat16": False,
            "seed": int(wrapper_cfg.get("seed", 42)),
            "root": ROOT_DIR,
            "plotter": {
                "save_video": False,
                "save_frames": False,
                "gif_duration": 1000,
                "plotter_resolution": 1000,
                "plotter_fig_size": 10,
            },
            "logger": {
                "loggers": ["console", "file"],
                "level": "INFO",
                "run_id": "wrapper",
            },
            "model": {
                "name": str(wrapper_cfg.get("model_name", "gpt-4o-mini")),
                "tokenizer_pad": "\\[PAD\\]",
                "tokenizer_padding_side": "left",
                "visual": False,
                "cache_dir": "",
                "seed_function_prompt": "seed_functions/generate_seed.txt",
                "max_new_tokens": int(wrapper_cfg.get("max_new_tokens", 2048)),
                "top_p": float(wrapper_cfg.get("top_p", 0.9)),
                "top_k": int(wrapper_cfg.get("top_k", 60)),
                "num_beams": int(wrapper_cfg.get("num_beams", 1)),
                "temperature": float(wrapper_cfg.get("temperature", 1.0)),
                "temperature_schedule": bool(wrapper_cfg.get("temperature_schedule", False)),
                "temperature_schedule_gamma": float(wrapper_cfg.get("temperature_schedule_gamma", 0.995)),
                "base_prompt": {
                    "prompt": "basic_text.txt",
                    "prompt_size": int(wrapper_cfg.get("prompt_size", 5)),
                },
            },
            "experiment": {
                "generate_seed_functions": True,
                "optimizer": {
                    "optimizer_threads": int(wrapper_cfg.get("optimizer_threads", 5)),
                    "timeout": int(wrapper_cfg.get("optimizer_timeout", 10)),
                    "p0_min": float(wrapper_cfg.get("optimizer_p0_min", -5)),
                    "p0_max": float(wrapper_cfg.get("optimizer_p0_max", 5)),
                    "coeff_rounding": int(wrapper_cfg.get("optimizer_coeff_rounding", 4)),
                    "tol": float(wrapper_cfg.get("optimizer_tol", 1e-3)),
                },
                "seed_functions": {
                    "functions": [],
                    "max_tries": int(wrapper_cfg.get("seed_generation_max_tries", 10)),
                    "generation_tokens": int(wrapper_cfg.get("seed_generation_tokens", 512)),
                },
                "scorer": {
                    "name": str(wrapper_cfg.get("scorer_name", "complexity_scorer")),
                    "rounding": int(wrapper_cfg.get("scorer_rounding", 8)),
                    "scientific": bool(wrapper_cfg.get("scorer_scientific", False)),
                    "normalize": bool(wrapper_cfg.get("scorer_normalize", False)),
                    "lambda": float(wrapper_cfg.get("scorer_lambda", 0.05)),
                    "max_nodes": int(wrapper_cfg.get("scorer_max_nodes", 30)),
                    "alternative": bool(wrapper_cfg.get("scorer_alternative", False)),
                },
                "function": {
                    "name": f"func{function_id}",
                    "group": "project_benchmark",
                    "train_points": {
                        "generate_points": False,
                        "data_folder": data_folder_rel.replace("\\", "/"),
                    },
                    "test_points": {
                        "min_points": 0,
                        "max_points": 1,
                        "num_points": 0,
                    },
                    "tolerance": float(wrapper_cfg.get("tolerance", 0.99999)),
                    "num_variables": int(meta["n_variables"]),
                    "iterations": int(wrapper_cfg.get("iterations", 10)),
                },
            },
        }
    )


def save_points(data_dir, x_train, y_train, x_test, y_test):
    train_points = np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1)
    test_points = np.concatenate([x_test, y_test.reshape(-1, 1)], axis=1)
    np.save(os.path.join(data_dir, "train_points.npy"), train_points)
    np.save(os.path.join(data_dir, "test_points.npy"), test_points)


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def evaluate_best_function(icsr_utils, best_function_str, x_train, y_train, x_test, y_test, n_variables):
    fn = icsr_utils.string_to_function(best_function_str, int(n_variables))
    train_pred = np.asarray(icsr_utils.eval_function(fn, x_train, int(n_variables))).reshape(-1)
    test_pred = np.asarray(icsr_utils.eval_function(fn, x_test, int(n_variables))).reshape(-1)
    return mse(y_train, train_pred), mse(y_test, test_pred)


def train_one(function_id, metadata, wrapper_cfg):
    api_key, api_base_url = prepare_api_environment(wrapper_cfg)
    if not api_key:
        raise RuntimeError(
            "ICSR uses the OpenAI model path by default, but no API key was found in "
            "`OPENAI_API_KEY`, `API_KEY`, or `DASHSCOPE_API_KEY`."
        )

    Workspace, icsr_utils = import_icsr_modules()
    patch_openai_model()
    patch_utils_load_model(icsr_utils, wrapper_cfg)
    dataset_root = resolve_path(wrapper_cfg["DATASET_PATH"])
    output_root = resolve_path(wrapper_cfg["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    raw_input_dir = os.path.join(func_output_dir, "raw_inputs")
    ensure_output_dir(raw_input_dir)

    meta = metadata[function_id]
    x_train, y_train, x_test, y_test, train_path, test_path = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=meta["n_variables"],
    )
    save_points(raw_input_dir, x_train, y_train, x_test, y_test)

    cfg = build_cfg(
        wrapper_cfg=wrapper_cfg,
        function_id=function_id,
        meta=meta,
        data_folder_rel=to_repo_relative(raw_input_dir),
    )
    cfg_path = os.path.join(func_output_dir, "config.generated.yaml")
    ensure_output_dir(func_output_dir)
    with open(cfg_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    start = time.time()
    workspace = Workspace(cfg)
    workspace.run()
    runtime_seconds = time.time() - start

    raw_run_dir = workspace.output_path
    results_path = os.path.join(raw_run_dir, "results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    best_function = str(raw_results["best_function"])
    train_mse, test_mse = evaluate_best_function(
        icsr_utils=icsr_utils,
        best_function_str=best_function,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_variables=meta["n_variables"],
    )

    result = {
        "algorithm": "ICSR",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "train_path": train_path,
        "test_path": test_path,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "runtime_seconds": runtime_seconds,
        "best_expression": best_function,
        "best_expr": str(raw_results.get("best_expr", "")),
        "best_found_at": int(raw_results.get("best_found_at", 0)),
        "test_score": raw_results.get("test_score"),
        "r2_train": raw_results.get("r2_train"),
        "r2_test": raw_results.get("r2_test"),
        "r2_all": raw_results.get("r2_all"),
        "final_complexity": raw_results.get("final_complexity"),
        "raw_run_dir": raw_run_dir,
        "generated_config_path": cfg_path,
        "api_base_url": api_base_url,
        "status": "success",
    }

    result = apply_rag_sr_result_schema(result)
    write_json(os.path.join(func_output_dir, "run_summary.json"), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ICSR baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "icsr", "config.yml"),
        help="Path to ICSR wrapper config YAML.",
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
            print(f"[ICSR] Running func{function_id} ({metadata[function_id]['name']})")
            result = train_one(function_id=function_id, metadata=metadata, wrapper_cfg=wrapper_cfg)
            all_results.append(result)
            print(
                f"[ICSR] func{function_id} done | "
                f"train_mse={result['train_mse']:.6g} | "
                f"test_mse={result['test_mse']:.6g} | "
                f"time={result['runtime_seconds']:.2f}s"
            )
    except RuntimeError as exc:
        raise SystemExit(f"[ICSR] {exc}") from exc

    append_jsonl(summary_path, all_results)
    print(f"[ICSR] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
