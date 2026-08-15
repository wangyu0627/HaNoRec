"""Derive HaNoRec preference data and offline HaRS artifacts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["microlens", "netflix", "movielens"])
    parser.add_argument("--hit", required=True, type=int, choices=[1, 3])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=str(ROOT / "Qwen-2.5-VL-3B-Instruct"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--preferences-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    common = [
        "--dataset",
        args.dataset,
        "--hit",
        str(args.hit),
        "--seed",
        str(args.seed),
        "--repo-root",
        str(ROOT),
    ]
    preference_command = [
        sys.executable,
        "-m",
        "hanorec.cli.build_preferences",
        *common,
    ]
    if args.force:
        preference_command.append("--force")

    commands = [preference_command]
    if not args.preferences_only:
        hardness_command = [
            sys.executable,
            "-m",
            "hanorec.cli.build_hardness",
            *common,
            "--top-k",
            str(args.top_k),
            "--model",
            args.model,
            "--device-map",
            args.device_map,
        ]
        if args.force:
            hardness_command.append("--force")
        commands.append(hardness_command)
    return commands


def main() -> None:
    args = parse_args()
    for command in build_commands(args):
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
