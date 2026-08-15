"""Build preference data from the repository's current SFT files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanorec.data.preference import build_preference_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["microlens", "netflix", "movielens"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--hit", type=int, required=True, choices=[1, 3])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--titles", type=Path)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    data_dir = root / "data" / args.dataset
    source = args.source or data_dir / f"{args.split}-mllm_sft_{args.hit}.json"
    title_name = {
        "microlens": "Microlens_titles.csv",
        "netflix": "Netflix_titles.csv",
        "movielens": "Movielens_titles.csv",
    }[args.dataset]
    titles = args.titles or data_dir / title_name
    tsv = args.tsv or data_dir / f"{args.split}.tsv"
    output = args.output or (
        root
        / "artifacts"
        / "hanorec"
        / args.dataset
        / f"{args.split}-mllm_dpo_{args.hit}.json"
    )
    report = build_preference_file(
        source,
        titles,
        tsv,
        output,
        hit=args.hit,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
