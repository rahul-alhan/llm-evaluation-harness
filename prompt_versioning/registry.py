"""File-backed prompt registry with hash-based versioning."""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path("prompts/registry.json")
PROMPTS_DIR = Path("prompts")


@dataclass
class PromptVersion:
    name: str
    version: int
    hash: str
    author: str
    created_at: str
    body_path: str


def _load() -> list[dict]:
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text("[]")
    return json.loads(REGISTRY_PATH.read_text())


def _save(entries: list[dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def register(name: str, file: str, author: str) -> PromptVersion:
    body = Path(file).read_text(encoding="utf-8")
    h = _hash(body)
    entries = _load()

    if any(e["name"] == name and e["hash"] == h for e in entries):
        raise ValueError(f"Prompt '{name}' with identical content already registered.")

    versions = [e["version"] for e in entries if e["name"] == name]
    next_v = max(versions, default=0) + 1

    body_path = PROMPTS_DIR / f"{name}__v{next_v}__{h}.txt"
    body_path.write_text(body, encoding="utf-8")

    pv = PromptVersion(
        name=name,
        version=next_v,
        hash=h,
        author=author,
        created_at=datetime.now(timezone.utc).isoformat(),
        body_path=str(body_path),
    )
    entries.append(asdict(pv))
    _save(entries)
    return pv


def load(name: str, version: int | None = None) -> str:
    entries = [e for e in _load() if e["name"] == name]
    if not entries:
        raise KeyError(f"No prompt registered with name '{name}'")
    if version is None:
        entry = max(entries, key=lambda e: e["version"])
    else:
        entry = next(e for e in entries if e["version"] == version)
    return Path(entry["body_path"]).read_text(encoding="utf-8")


def diff(name: str, va: int, vb: int) -> str:
    a = load(name, va).splitlines(keepends=True)
    b = load(name, vb).splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=f"{name}@v{va}", tofile=f"{name}@v{vb}"))


def list_all() -> list[dict]:
    return _load()
