import os
from pathlib import Path


_ENV_LOADED = False
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _set_env_value(key: str, value: str, override: bool = False):
    if value and (override or not os.environ.get(key)):
        os.environ[key] = value


def _apply_env_aliases(override: bool = False):
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    qwen_base_url = os.environ.get("QWEN_BASE_URL")
    qwen_model = os.environ.get("QWEN_MODEL")

    _set_env_value("API_KEY", dashscope_key, override=override)
    _set_env_value("OPENAI_API_KEY", dashscope_key, override=override)
    _set_env_value("OPENAI_BASE_URL", qwen_base_url, override=override)
    _set_env_value("OPENAI_MODEL", qwen_model, override=override)


def resolve_api_key(*env_names: str):
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def resolve_api_base_url(config_value=None, default: str = DEFAULT_OPENAI_BASE_URL, *env_names: str):
    if config_value:
        return str(config_value).rstrip("/")
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return str(value).rstrip("/")
    return default.rstrip("/")


def load_env_file(env_path=None, override=True) -> bool:
    global _ENV_LOADED
    if _ENV_LOADED and not override:
        return True

    path = Path(env_path) if env_path else _repo_root() / ".env"
    if not path.exists():
        _ENV_LOADED = True
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = _strip_quotes(value)

    _apply_env_aliases(override=override)
    _ENV_LOADED = True
    return True
