"""Prompt registry smoke tests.

Covers the provenance contract: register → resolve returns the entry,
load returns the body, latest version wins when version is None, and
duplicate registration is rejected.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def registry_in_tmp(tmp_path, monkeypatch):
    """Run the registry against an isolated prompts/ directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    # Re-import the module fresh so it picks up the new cwd
    if "prompt_versioning.registry" in sys.modules:
        del sys.modules["prompt_versioning.registry"]
    return importlib.import_module("prompt_versioning.registry")


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_register_creates_v1(registry_in_tmp, tmp_path):
    src = _write(tmp_path, "p.txt", "Be terse.")
    pv = registry_in_tmp.register(name="terse", file=src, author="rahul")
    assert pv.version == 1
    assert pv.author == "rahul"
    assert pv.hash and len(pv.hash) == 12


def test_resolve_latest_when_version_is_none(registry_in_tmp, tmp_path):
    registry_in_tmp.register(name="t", file=_write(tmp_path, "a.txt", "v1 body"), author="r")
    registry_in_tmp.register(name="t", file=_write(tmp_path, "b.txt", "v2 body"), author="r")

    entry = registry_in_tmp.resolve("t", version=None)
    assert entry["version"] == 2
    assert "hash" in entry and entry["hash"]


def test_load_returns_correct_version_body(registry_in_tmp, tmp_path):
    registry_in_tmp.register(name="t", file=_write(tmp_path, "a.txt", "v1 body"), author="r")
    registry_in_tmp.register(name="t", file=_write(tmp_path, "b.txt", "v2 body"), author="r")

    assert registry_in_tmp.load("t", version=1) == "v1 body"
    assert registry_in_tmp.load("t", version=2) == "v2 body"
    assert registry_in_tmp.load("t") == "v2 body"  # default = latest


def test_register_rejects_exact_duplicate(registry_in_tmp, tmp_path):
    src = _write(tmp_path, "p.txt", "Same body.")
    registry_in_tmp.register(name="x", file=src, author="r")
    with pytest.raises(ValueError, match="already registered"):
        registry_in_tmp.register(name="x", file=src, author="r")


def test_resolve_unknown_name_raises(registry_in_tmp):
    with pytest.raises(KeyError):
        registry_in_tmp.resolve("never-registered")


def test_diff_between_versions(registry_in_tmp, tmp_path):
    registry_in_tmp.register(name="t", file=_write(tmp_path, "a.txt", "alpha\nbeta\n"), author="r")
    registry_in_tmp.register(name="t", file=_write(tmp_path, "b.txt", "alpha\ngamma\n"), author="r")
    d = registry_in_tmp.diff("t", 1, 2)
    assert "-beta" in d
    assert "+gamma" in d
