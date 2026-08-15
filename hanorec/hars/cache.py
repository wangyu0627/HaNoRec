"""Fingerprint and atomic-write helpers for derived HaNoRec artifacts."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class StaleArtifactError(ValueError):
    """Raised when a derived artifact does not match its current inputs."""


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    stat = source.stat()
    return {
        "path": source.as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def directory_fingerprint(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    digest = hashlib.sha256()
    count = 0
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        stat = file_path.stat()
        relative = file_path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        count += 1
    return {
        "path": root.as_posix(),
        "files": count,
        "metadata_sha256": digest.hexdigest(),
    }


def build_embedding_manifest(
    *,
    dataset: str,
    model_id: str,
    title_path: str | Path,
    image_dir: str | Path,
    extra_item_ids: Iterable[int] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "catalog_embeddings",
        "dataset": dataset,
        "model_id": model_id,
        "title": file_fingerprint(title_path),
        "images": directory_fingerprint(image_dir),
        "extra_item_ids": sorted(set(int(item_id) for item_id in extra_item_ids)),
    }


def build_manifest(
    *,
    dataset: str,
    model_id: str,
    input_files: list[str | Path],
    top_k: int,
    seed: int,
    missing_images: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if missing_images < 0:
        raise ValueError("missing_images must be non-negative")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model_id": model_id,
        "top_k": int(top_k),
        "seed": int(seed),
        "missing_images": int(missing_images),
        "inputs": [file_fingerprint(path) for path in input_files],
    }


def validate_manifest(cached: dict[str, Any], expected: dict[str, Any]) -> None:
    if cached == expected:
        return
    keys = sorted(set(cached) | set(expected))
    changed = [key for key in keys if cached.get(key) != expected.get(key)]
    raise StaleArtifactError(
        "HaNoRec artifact is stale; changed manifest fields: " + ", ".join(changed)
    )


def atomic_json_dump(data: Any, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_npz_dump(output_path: str | Path, **arrays: Any) -> None:
    import numpy as np

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
