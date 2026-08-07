"""Loads each sibling repo's own .env rather than duplicating credentials
into riskdesk's environment. All six source repos live as sibling
directories on this machine; SIBLINGS_ROOT can be overridden via
RISKDESK_SIBLINGS_ROOT for a different checkout layout (e.g. CI).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

SIBLINGS_ROOT = Path(os.environ.get("RISKDESK_SIBLINGS_ROOT", Path.home()))


def repo_path(repo_dir_name: str) -> Path:
    return SIBLINGS_ROOT / repo_dir_name


def load_repo_env(repo_dir_name: str) -> dict[str, str | None]:
    """Reads <repo>/.env directly. Returns {} if the repo or its .env is absent."""
    env_path = repo_path(repo_dir_name) / ".env"
    if not env_path.exists():
        return {}
    return dotenv_values(env_path)
