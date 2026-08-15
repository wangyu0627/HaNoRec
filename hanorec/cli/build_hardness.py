"""Encode the catalog and attach paper-aligned HaRS hardness values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hanorec.hars.cache import (
    atomic_json_dump,
    build_embedding_manifest,
    build_manifest,
    load_json,
    validate_manifest,
)
from hanorec.hars.encoder import encode_catalog
from hanorec.hars.hardness import attach_hardness, compute_sample_hardness, topk_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["microlens", "netflix", "movielens"])
    parser.add_argument("--hit", type=int, required=True, choices=[1, 3])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--model", default="Qwen-2.5-VL-3B-Instruct")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--titles", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--neighbors", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    artifact_dir = root / "artifacts" / "hanorec" / args.dataset
    data_dir = root / "data" / args.dataset
    title_name = {
        "microlens": "Microlens_titles.csv",
        "netflix": "Netflix_titles.csv",
        "movielens": "Movielens_titles.csv",
    }[args.dataset]
    preferences_path = args.preferences or artifact_dir / f"{args.split}-mllm_dpo_{args.hit}.json"
    title_path = args.titles or data_dir / title_name
    image_dir = args.images or data_dir / "images"
    embeddings_path = args.embeddings or artifact_dir / "catalog_embeddings.npz"
    embeddings_manifest_path = embeddings_path.with_suffix(".manifest.json")
    neighbors_path = args.neighbors or artifact_dir / f"neighbors-k{args.top_k}.json"
    output_path = args.output or artifact_dir / f"{args.split}-mllm_hanorec_{args.hit}.json"
    manifest_path = output_path.with_suffix(".manifest.json")

    for path in (neighbors_path, output_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    examples = load_json(preferences_path)
    if not isinstance(examples, list) or not examples:
        raise ValueError("Preference file must contain a non-empty JSON list")
    referenced_ids = {
        int(item_id)
        for example in examples
        for key in ("positive_item_ids", "negative_item_ids")
        for item_id in example.get(key, [])
    }
    expected_embeddings_manifest = build_embedding_manifest(
        dataset=args.dataset,
        model_id=args.model,
        title_path=title_path,
        image_dir=image_dir,
        extra_item_ids=referenced_ids,
    )

    if not embeddings_path.exists() or args.force:
        encode_report = encode_catalog(
            title_path=title_path,
            image_dir=image_dir,
            model_name_or_path=args.model,
            output_path=embeddings_path,
            extra_item_ids=referenced_ids,
            device_map=args.device_map,
        )
        atomic_json_dump(expected_embeddings_manifest, embeddings_manifest_path)
    else:
        if not embeddings_manifest_path.is_file():
            raise FileNotFoundError(
                f"Embedding manifest missing: {embeddings_manifest_path}; rerun with --force."
            )
        validate_manifest(load_json(embeddings_manifest_path), expected_embeddings_manifest)
        encode_report = {
            "output": str(embeddings_path), "manifest": str(embeddings_manifest_path), "reused": True
        }

    encode_report.setdefault("manifest", str(embeddings_manifest_path))
    with np.load(embeddings_path) as cache:
        item_ids = cache["item_ids"]
        fused_embeddings = cache["fused_embeddings"]
        missing_images = int(cache["missing_images"][0])
    neighbors = topk_neighbors(
        item_ids,
        fused_embeddings,
        k=args.top_k,
        block_size=args.block_size,
    )
    hardness = compute_sample_hardness(examples, neighbors)
    attached = attach_hardness(examples, hardness)

    serializable_neighbors = {str(item_id): value for item_id, value in neighbors.items()}
    atomic_json_dump(serializable_neighbors, neighbors_path)
    atomic_json_dump(attached, output_path)
    manifest = build_manifest(
        dataset=args.dataset,
        model_id=args.model,
        input_files=[preferences_path, title_path, embeddings_path],
        top_k=args.top_k,
        seed=args.seed,
        missing_images=missing_images,
    )
    atomic_json_dump(manifest, manifest_path)
    print(
        json.dumps(
            {
                "samples": len(attached),
                "mean_hardness": float(np.mean(hardness)),
                "min_hardness": float(np.min(hardness)),
                "max_hardness": float(np.max(hardness)),
                "embeddings": encode_report,
                "neighbors": str(neighbors_path),
                "output": str(output_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
