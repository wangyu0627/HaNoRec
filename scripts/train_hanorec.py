"""Prepare current repository data and launch HaNoRec DPO training."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["microlens", "netflix", "movielens"])
    parser.add_argument("--hit", required=True, type=int, choices=[1, 3])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=str(ROOT / "Qwen-2.5-VL-3B-Instruct"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _python_path_environment() -> dict[str, str]:
    env = os.environ.copy()
    required = [str(ROOT), str(ROOT / "LLaMA-Factory" / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        required.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(required)
    return env


def main() -> None:
    args = parse_args()
    if not args.skip_prepare:
        prepare_command = [
            sys.executable,
            str(ROOT / "scripts" / "prepare_hanorec.py"),
            "--dataset",
            args.dataset,
            "--hit",
            str(args.hit),
            "--top-k",
            str(args.top_k),
            "--model",
            args.model,
            "--device-map",
            args.device_map,
            "--seed",
            str(args.seed),
        ]
        if args.force_prepare:
            prepare_command.append("--force")
        if args.dry_run:
            prepare_command.append("--dry-run")
        print(shlex.join(prepare_command))
        if not args.dry_run:
            subprocess.run(prepare_command, cwd=ROOT, check=True)

    if args.prepare_only:
        return

    config = (
        args.config
        or ROOT / "configs" / "hanorec" / f"{args.dataset}_hit{args.hit}.yaml"
    ).resolve()
    if not config.is_file():
        raise FileNotFoundError(f"HaNoRec config not found: {config}")
    train_command = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        str(config),
    ]
    print(shlex.join(train_command))
    if not args.dry_run:
        subprocess.run(
            train_command,
            cwd=ROOT,
            env=_python_path_environment(),
            check=True,
        )


if __name__ == "__main__":
    main()
