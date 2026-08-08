"""Loads each sibling repo's own .env rather than duplicating credentials
into riskdesk's environment. All six source repos live as sibling
directories on this machine; SIBLINGS_ROOT can be overridden via
RISKDESK_SIBLINGS_ROOT for a different checkout layout (e.g. CI).

Deployment note: a deployed backend (Render) has no local filesystem
access to sibling repos' .env files at all -- there is no "sibling
directory" on that machine. `load_repo_env` therefore also accepts a
direct environment-variable override per key, prefixed with the repo's
own slug (hyphens -> underscores, upper-cased), e.g.
`ALPHA_SIGNAL_LAB_DATABASE_URL` overrides alpha-signal-lab's own
DATABASE_URL. The env var always wins over the local file when both are
present -- this is the same "explicit config beats an implicit local
file" convention already used for ALLOWED_ORIGINS elsewhere in this
project, not a new pattern invented just for deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

def siblings_root() -> Path:
    """Read fresh on every call, deliberately -- not a module-level
    constant. A constant computed once at import time would freeze
    whatever RISKDESK_SIBLINGS_ROOT happened to be at that moment, making
    it impossible to override afterward (in tests, or any other context
    that sets the env var after this module first loads).
    """
    return Path(os.environ.get("RISKDESK_SIBLINGS_ROOT", Path.home()))


def repo_path(repo_dir_name: str) -> Path:
    return siblings_root() / repo_dir_name


def load_repo_env(repo_dir_name: str) -> dict[str, str | None]:
    """Reads <repo>/.env if present, then overlays any `{PREFIX}_{KEY}`
    environment variables (see module docstring) -- the only path that
    works once this project is deployed somewhere without local sibling
    checkouts at all.
    """
    env_path = repo_path(repo_dir_name) / ".env"
    local = dict(dotenv_values(env_path)) if env_path.exists() else {}

    prefix = f"{repo_dir_name.upper().replace('-', '_')}_"
    overrides = {
        key[len(prefix):]: value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }
    return {**local, **overrides}
