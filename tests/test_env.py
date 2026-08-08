"""Tests for connectors/_env.py's deployment override mechanism -- the
piece that lets the backend run with zero local filesystem access to the
six sibling repos (exactly what a deployed Render instance sees), by
supplying values via prefixed environment variables instead of a sibling's
local .env file.
"""

from __future__ import annotations

import pytest

from connectors._env import load_repo_env


def test_env_var_override_works_with_no_local_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))  # no sibling checkouts here at all
    monkeypatch.setenv("ALPHA_SIGNAL_LAB_DATABASE_URL", "postgresql://from-env-var")

    env = load_repo_env("alpha-signal-lab")
    assert env.get("DATABASE_URL") == "postgresql://from-env-var"


def test_env_var_override_takes_precedence_over_local_file(monkeypatch, tmp_path):
    repo_dir = tmp_path / "alpha-signal-lab"
    repo_dir.mkdir()
    (repo_dir / ".env").write_text("DATABASE_URL=postgresql://from-local-file\n")
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_SIGNAL_LAB_DATABASE_URL", "postgresql://from-env-var")

    env = load_repo_env("alpha-signal-lab")
    assert env.get("DATABASE_URL") == "postgresql://from-env-var"


def test_local_file_used_when_no_override_present(monkeypatch, tmp_path):
    repo_dir = tmp_path / "alpha-signal-lab"
    repo_dir.mkdir()
    (repo_dir / ".env").write_text("DATABASE_URL=postgresql://from-local-file\n")
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))
    monkeypatch.delenv("ALPHA_SIGNAL_LAB_DATABASE_URL", raising=False)

    env = load_repo_env("alpha-signal-lab")
    assert env.get("DATABASE_URL") == "postgresql://from-local-file"


def test_prefix_matching_is_repo_specific_not_a_substring_match(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))
    monkeypatch.setenv("STREAMALPHA_DATABASE_URL", "postgresql://streamalpha-db")
    # A different repo's override must not leak into this one's lookup.
    env = load_repo_env("alpha-signal-lab")
    assert env.get("DATABASE_URL") is None


def test_returns_empty_dict_when_nothing_present(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))
    env = load_repo_env("a-repo-with-no-env-and-no-overrides")
    assert env == {}


def test_end_to_end_registry_works_with_zero_local_sibling_access(monkeypatch, tmp_path):
    """The actual deployment scenario: point RISKDESK_SIBLINGS_ROOT
    somewhere with nothing in it, supply only the documented env vars
    (render.yaml's list), and confirm the same real book comes back.
    """
    import os

    real_alpha_db = None
    real_stream_db = None
    real_apca_key = None
    real_apca_secret = None
    home = os.environ.get("RISKDESK_SIBLINGS_ROOT") or str(__import__("pathlib").Path.home())
    from dotenv import dotenv_values

    alpha_env = dotenv_values(f"{home}/alpha-signal-lab/.env")
    stream_env = dotenv_values(f"{home}/streamalpha/.env")
    real_alpha_db = alpha_env.get("DATABASE_URL")
    real_stream_db = stream_env.get("DATABASE_URL")
    real_apca_key = alpha_env.get("ALPACA_API_KEY")
    real_apca_secret = alpha_env.get("ALPACA_SECRET_KEY")

    if not all([real_alpha_db, real_stream_db, real_apca_key, real_apca_secret]):
        pytest.skip("Real sibling credentials not available in this environment to simulate deployment.")

    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_SIGNAL_LAB_DATABASE_URL", real_alpha_db)
    monkeypatch.setenv("STREAMALPHA_DATABASE_URL", real_stream_db)
    monkeypatch.setenv("APCA_API_KEY_ID", real_apca_key)
    monkeypatch.setenv("APCA_API_SECRET_KEY", real_apca_secret)

    from connectors.registry import fetch_all

    positions, meta = fetch_all()
    assert len(positions) == 33
    alpha_meta = next(m for m in meta if m.repo == "alpha-signal-lab")
    assert alpha_meta.n_positions == 10
    bookmaker_meta = next(m for m in meta if m.repo == "bookmaker")
    assert "stand-in" in bookmaker_meta.notes.lower()
