"""Path resolution for the self-contained baselines directory.

External dependencies (external/*, iclrcom/*, .pysr_runtime) live inside
this directory alongside wrappers/, results/, etc.

Override the root with the ``QWEN_DEPS_ROOT`` environment variable.
"""

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_deps_root():
    """Absolute path to the dependency root (this directory)."""
    user_deps = os.environ.get("QWEN_DEPS_ROOT")
    if user_deps:
        return os.path.abspath(user_deps)
    return _PROJECT_ROOT


def resolve_deps_path(*segments):
    """Shortcut: ``resolve_deps_path("external", "SR-LLM")``."""
    return os.path.join(get_deps_root(), *segments)


def resolve_project_path(*segments):
    """Shortcut: path relative to this project root."""
    return os.path.join(_PROJECT_ROOT, *segments)
