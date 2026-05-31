import argparse
import http.client
import json
import os
import sys
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.env_loader import load_env_file, resolve_api_base_url, resolve_api_key
from utils.baseline_result_schema import apply_rag_sr_result_schema
from utils.paths import resolve_deps_path
from utils.readAndwrite import append_jsonl, write_json

UPSTREAM_ROOT = resolve_deps_path("external", "LLM-SR")
if UPSTREAM_ROOT not in sys.path:
    sys.path.insert(0, UPSTREAM_ROOT)


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


def load_xy(dataset_root, function_id, n_variables, meta_entry=None):
    from utils.data_loader import resolve_evosr, load_data as _ld
    fp_e, inp, tgt = resolve_evosr(meta_entry) if meta_entry else (None, None, None)
    if fp_e:
        return _ld(fp_e, inp, tgt) + (fp_e["train_data"], fp_e["test_data"])
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
            f"LLM-SR wrapper currently supports 1D/2D benchmark tasks, got n_variables={n_variables}."
        )

    x_train = df_train[feature_columns].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    x_test = df_test[feature_columns].to_numpy(dtype=float)
    y_test = df_test["y"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, train_path, test_path


def import_llmsr_modules():
    try:
        from llmsr import pipeline
        from llmsr import config as config_lib
        from llmsr import sampler
        from llmsr import evaluator
        from llmsr import code_manipulation
    except Exception as exc:
        raise RuntimeError(
            "LLM-SR wrapper found the upstream source tree, but could not import its runtime "
            "dependencies. The active environment likely still needs packages from "
            "`baselines/external/LLM-SR/requirements.txt` (see --deps-dir)."
        ) from exc
    return pipeline, config_lib, sampler, evaluator, code_manipulation


def prepare_api_environment(cfg):
    load_env_file()

    api_key = resolve_api_key("API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")
    api_base_url = resolve_api_base_url(
        cfg.get("api_base_url"),
        "https://api.openai.com/v1",
        "OPENAI_BASE_URL",
        "QWEN_BASE_URL",
    )
    os.environ["LLM_SR_API_BASE_URL"] = api_base_url
    return api_key, os.environ["LLM_SR_API_BASE_URL"]


def patch_sampler_api(sampler):
    def _draw_samples_api(self, prompt, config):
        all_samples = []
        prompt = "\n".join([self._instruction_prompt, prompt])
        base_url = os.environ.get("LLM_SR_API_BASE_URL", "https://api.openai.com/v1")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError(f"Unsupported api_base_url scheme for LLM-SR: {base_url}")

        request_path = parsed.path.rstrip("/") + "/chat/completions"
        if not request_path.startswith("/"):
            request_path = "/" + request_path

        connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        port = parsed.port

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = connection_cls(parsed.hostname, port, timeout=60)
                    payload = json.dumps(
                        {
                            "max_tokens": 512,
                            "model": config.api_model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                        }
                    )
                    headers = {
                        "Authorization": f"Bearer {os.environ['API_KEY']}",
                        "User-Agent": "LLM-SR-Wrapper/1.0",
                        "Content-Type": "application/json",
                    }
                    conn.request("POST", request_path, payload, headers)
                    res = conn.getresponse()
                    raw_data = res.read().decode("utf-8")
                    data = json.loads(raw_data)
                    response = data["choices"][0]["message"]["content"]

                    if self._trim:
                        response = sampler._extract_body(response, config)

                    all_samples.append(response)
                    break
                except Exception:
                    continue
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

        return all_samples

    sampler.LocalLLM._draw_samples_api = _draw_samples_api


def build_specification(n_variables, max_nparams):
    if int(n_variables) == 1:
        prompt = (
            "Find the mathematical function that maps x1 to y given observational data."
        )
        args_decl = "x1: np.ndarray, params: np.ndarray"
        unpack = "    x1 = inputs[:, 0]"
        equation_call = "equation(x1, params)"
        arg_docs = (
            "        x1: Observations of the first input variable.\n"
            "        params: Array of numeric constants to be optimized.\n"
        )
        initial_body = "    y = params[0] * x1 + params[1]\n    return y"
    else:
        prompt = (
            "Find the mathematical function that maps x1 and x2 to y given observational data."
        )
        args_decl = "x1: np.ndarray, x2: np.ndarray, params: np.ndarray"
        unpack = "    x1, x2 = inputs[:, 0], inputs[:, 1]"
        equation_call = "equation(x1, x2, params)"
        arg_docs = (
            "        x1: Observations of the first input variable.\n"
            "        x2: Observations of the second input variable.\n"
            "        params: Array of numeric constants to be optimized.\n"
        )
        initial_body = "    y = params[0] * x1 + params[1] * x2 + params[2]\n    return y"

    return f'''"""
{prompt}
"""

import numpy as np
from scipy.optimize import minimize

MAX_NPARAMS = {int(max_nparams)}
params = [1.0] * MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on data observations."""
    inputs, outputs = data["inputs"], data["outputs"]
{unpack}

    def loss(params):
        y_pred = {equation_call}
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0] * MAX_NPARAMS, method="BFGS")
    loss_value = result.fun
    if np.isnan(loss_value) or np.isinf(loss_value):
        return None
    return -loss_value


@equation.evolve
def equation({args_decl}) -> np.ndarray:
    """Mathematical function to be evolved for symbolic regression.

    Args:
{arg_docs}    Return:
        Predicted target values.
    """
{initial_body}
'''


def build_config(config_lib, cfg):
    exp_cfg = config_lib.ExperienceBufferConfig(
        functions_per_prompt=int(cfg.get("functions_per_prompt", 2)),
        num_islands=int(cfg.get("num_islands", 8)),
        reset_period=int(cfg.get("reset_period", 14400)),
        cluster_sampling_temperature_init=float(cfg.get("cluster_sampling_temperature_init", 0.1)),
        cluster_sampling_temperature_period=int(cfg.get("cluster_sampling_temperature_period", 30000)),
    )
    return config_lib.Config(
        experience_buffer=exp_cfg,
        num_samplers=int(cfg.get("num_samplers", 1)),
        num_evaluators=int(cfg.get("num_evaluators", 1)),
        samples_per_prompt=int(cfg.get("samples_per_prompt", 4)),
        evaluate_timeout_seconds=int(cfg.get("evaluate_timeout_seconds", 30)),
        use_api=bool(cfg.get("use_api", True)),
        api_model=str(cfg.get("api_model", "gpt-4o-mini")),
    )


def load_best_sample(samples_dir):
    best = None
    for filename in os.listdir(samples_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(samples_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        score = row.get("score")
        if score is None:
            continue
        if best is None or float(score) > float(best["score"]):
            best = row
    if best is None:
        raise RuntimeError(
            "LLM-SR finished without any successful evaluated samples. "
            "Please inspect the raw sample logs for runtime or API failures."
        )
    return best


def mse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


def fit_best_program(function_code, specification, x_train, y_train, x_test, y_test, max_nparams, code_manipulation):
    func = code_manipulation.text_to_function(function_code)
    template = code_manipulation.text_to_program(specification)
    template.get_function("equation").body = func.body
    program = str(template)

    globals_dict = {"minimize": minimize}
    exec(program, globals_dict)
    equation = globals_dict["equation"]

    if x_train.shape[1] == 1:
        def predict(x, params):
            return np.asarray(equation(x[:, 0], params), dtype=float).reshape(-1)
    else:
        def predict(x, params):
            return np.asarray(equation(x[:, 0], x[:, 1], params), dtype=float).reshape(-1)

    def objective(params):
        pred = predict(x_train, params)
        return np.mean((pred - y_train) ** 2)

    result = minimize(objective, [1.0] * int(max_nparams), method="BFGS")
    params = np.asarray(result.x, dtype=float)
    train_pred = predict(x_train, params)
    test_pred = predict(x_test, params)
    return {
        "program": program,
        "optimized_params": params.tolist(),
        "train_mse": mse(y_train, train_pred),
        "test_mse": mse(y_test, test_pred),
    }


def train_one(function_id, metadata, config):
    cfg = config  # alias for backward compatibility with function body
    # Force UTF-8 for stdout to prevent GBK crashes on Windows when upstream prints
    try:
        import sys, io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass
    pipeline, config_lib, sampler, evaluator, code_manipulation = import_llmsr_modules()
    api_key, api_base_url = prepare_api_environment(cfg)
    patch_sampler_api(sampler)
    if bool(cfg.get("use_api", True)) and not api_key:
        raise RuntimeError(
            "LLM-SR wrapper is configured with `use_api: true`, but no API key was found in "
            "`API_KEY`, `OPENAI_API_KEY`, or `DASHSCOPE_API_KEY`."
        )

    dataset_root = resolve_path(cfg["DATASET_PATH"])
    output_root = resolve_path(cfg["OUTPUT_PATH"])
    func_output_dir = os.path.join(output_root, f"func{function_id}")
    raw_log_dir = os.path.join(func_output_dir, str(cfg.get("raw_log_subdir", "raw_logs")))
    ensure_output_dir(raw_log_dir)

    meta = metadata[function_id]
    x_train, y_train, x_test, y_test, train_path, test_path = load_xy(
        dataset_root=dataset_root,
        function_id=function_id,
        n_variables=meta["n_variables"],
        meta_entry=meta,
    )

    specification = build_specification(meta["n_variables"], cfg.get("max_nparams", 10))
    spec_path = os.path.join(func_output_dir, "specification_generated.py")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(specification)

    llmsr_config = build_config(config_lib, cfg)
    class_config = config_lib.ClassConfig(
        llm_class=sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox,
    )
    inputs = {"data": {"inputs": x_train, "outputs": y_train}}

    start = time.time()
    pipeline.main(
        specification=specification,
        inputs=inputs,
        config=llmsr_config,
        max_sample_nums=int(cfg.get("max_sample_nums", 40)),
        class_config=class_config,
        log_dir=raw_log_dir,
    )
    runtime_seconds = time.time() - start

    samples_dir = os.path.join(raw_log_dir, "samples")
    best_sample = load_best_sample(samples_dir)
    fitted = fit_best_program(
        function_code=best_sample["function"],
        specification=specification,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        max_nparams=cfg.get("max_nparams", 10),
        code_manipulation=code_manipulation,
    )

    result = {
        "algorithm": "LLM-SR",
        "function_id": function_id,
        "function_name": meta["name"],
        "n_variables": int(meta["n_variables"]),
        "train_path": train_path,
        "test_path": test_path,
        "train_mse": fitted["train_mse"],
        "test_mse": fitted["test_mse"],
        "runtime_seconds": runtime_seconds,
        "best_expression": best_sample["function"],
        "best_score": float(best_sample["score"]),
        "best_sample_order": int(best_sample["sample_order"]),
        "optimized_params": fitted["optimized_params"],
        "spec_path": spec_path,
        "raw_log_dir": raw_log_dir,
        "status": "success",
        "api_base_url": api_base_url,
    }

    write_json(os.path.join(func_output_dir, "best_program.json"), best_sample)
    result = apply_rag_sr_result_schema(result)
    write_json(os.path.join(func_output_dir, "run_summary.json"), result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM-SR baseline on benchmark datasets.")
    parser.add_argument(
        "--config",
        default=os.path.join("wrappers", "llm_sr", "config.yml"),
        help="Path to LLM-SR wrapper config YAML.",
    )
    parser.add_argument("--function-id", type=int, default=None, help="Run one benchmark function.")
    parser.add_argument("--all", action="store_true", help="Run all benchmark functions.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config))
    metadata = load_metadata(resolve_path(cfg["BENCHMARK_METADATA_PATH"]))

    if args.all:
        function_ids = sorted(metadata.keys())
    elif args.function_id is not None:
        if args.function_id not in metadata:
            raise ValueError(f"Unknown function id: {args.function_id}")
        function_ids = [args.function_id]
    else:
        raise ValueError("Please provide --function-id N or use --all.")

    output_root = resolve_path(cfg["OUTPUT_PATH"])
    ensure_output_dir(output_root)
    summary_path = os.path.join(output_root, "summary.jsonl")

    all_results = []
    try:
        for function_id in function_ids:
            print(f"[LLM-SR] Running func{function_id} ({metadata[function_id]['name']})")
            result = train_one(function_id=function_id, metadata=metadata, cfg=cfg)
            all_results.append(result)
            print(
                f"[LLM-SR] func{function_id} done | "
                f"train_mse={result['train_mse']:.6g} | "
                f"test_mse={result['test_mse']:.6g} | "
                f"time={result['runtime_seconds']:.2f}s"
            )
    except RuntimeError as exc:
        raise SystemExit(f"[LLM-SR] {exc}") from exc

    append_jsonl(summary_path, all_results)
    print(f"[LLM-SR] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
