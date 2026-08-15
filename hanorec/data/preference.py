"""Derive HaNoRec pairwise data without rewriting current SFT files."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


_UNKNOWN_TITLE = re.compile(r"^\[unknown\s+(\d+)\]$", re.IGNORECASE)


class TitleIndex:
    """Resolve titles deterministically while retaining duplicate information."""

    def __init__(self, items: Iterable[tuple[int, str]]):
        self._item_to_title: dict[int, str] = {}
        title_to_items: dict[str, list[int]] = defaultdict(list)
        for raw_item_id, raw_title in items:
            item_id = int(raw_item_id)
            title = str(raw_title).strip()
            if title == "None":
                title = "nan"
            if not title:
                raise ValueError(f"Item {item_id} has an empty title")
            self._item_to_title[item_id] = title
            title_to_items[title].append(item_id)

        if not self._item_to_title:
            raise ValueError("Title index must be non-empty")
        self._title_to_items = {
            title: sorted(set(item_ids)) for title, item_ids in title_to_items.items()
        }
        self.duplicates = {
            title: item_ids for title, item_ids in self._title_to_items.items() if len(item_ids) > 1
        }

    @classmethod
    def from_csv(cls, path: str | Path) -> "TitleIndex":
        source = Path(path)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"item", "title"}.issubset(reader.fieldnames):
                raise ValueError(f"{source} must contain item and title columns")
            return cls((int(row["item"]), row["title"]) for row in reader)

    def _candidates(self, raw_title: str) -> tuple[str, list[int]]:
        title = str(raw_title).strip()
        item_ids = self._title_to_items.get(title)
        if item_ids:
            return title, item_ids

        title = _clean_title(title)
        unknown = _UNKNOWN_TITLE.match(title)
        if unknown:
            return title, [int(unknown.group(1))]
        return title, self._title_to_items.get(title, [])

    def resolve(self, raw_title: str, exclude: Iterable[int] = ()) -> int:
        title, item_ids = self._candidates(raw_title)
        excluded = {int(item_id) for item_id in exclude}
        eligible = [item_id for item_id in item_ids if item_id not in excluded]
        if not item_ids:
            raise ValueError(f"Unknown title: {title!r}")
        if not eligible:
            raise ValueError(
                f"Title {title!r} is unavailable after excluding {sorted(excluded)}"
            )
        return eligible[0]

    def matches(self, raw_title: str, item_id: int) -> bool:
        _, item_ids = self._candidates(raw_title)
        return int(item_id) in item_ids
    def title(self, item_id: int) -> str:
        return self._item_to_title.get(int(item_id), f"[unknown {int(item_id)}]")

    def items(self) -> list[tuple[int, str]]:
        return sorted(self._item_to_title.items())


def _clean_title(raw_line: str) -> str:
    value = raw_line.strip()
    if value.startswith("<image>"):
        value = value[len("<image>") :].strip()
    value = value.rstrip().rstrip(".").rstrip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1].strip()
    if not value:
        raise ValueError(f"Cannot parse title from line: {raw_line!r}")
    return value


def _message_pair(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("SFT sample must contain exactly one user and one assistant message")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("SFT sample roles must be user then assistant")
    if not isinstance(user.get("content"), str) or not isinstance(assistant.get("content"), str):
        raise ValueError("Message content must be text")
    return user, assistant


def _section_titles(content: str, start_text: str, end_text: str | None = None) -> list[str]:
    lines = content.splitlines()
    start = next(
        (index + 1 for index, line in enumerate(lines) if start_text.casefold() in line.casefold()),
        None,
    )
    if start is None:
        raise ValueError(f"Prompt section not found: {start_text!r}")

    result: list[str] = []
    for line in lines[start:]:
        if end_text and end_text.casefold() in line.casefold():
            break
        if not line.strip():
            continue
        result.append(_clean_title(line))
    if not result:
        raise ValueError(f"Prompt section {start_text!r} contains no titles")
    return result


def _target_title(content: str) -> str:
    return _section_titles(content, "Whether the user will like the target", None)[0]


def _candidate_titles(content: str) -> list[str]:
    return _section_titles(content, "Below are the candidate", None)


def _negative_context_titles(content: str) -> list[str]:
    return _section_titles(content, "not watched", "Whether the user will like the target")


def _base_pairwise_sample(
    sample: dict[str, Any],
    user: dict[str, Any],
    chosen: dict[str, Any],
    rejected_content: str,
    positive_item_ids: Sequence[int],
    negative_item_ids: Sequence[int],
) -> dict[str, Any]:
    images = sample.get("images") or []
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    return {
        "messages": [dict(user)],
        "chosen": dict(chosen),
        "rejected": {"role": "assistant", "content": rejected_content},
        "images": list(images),
        "positive_item_ids": [int(item_id) for item_id in positive_item_ids],
        "negative_item_ids": [int(item_id) for item_id in negative_item_ids],
    }


def derive_ranking_example(
    sample: dict[str, Any],
    title_index: TitleIndex,
    positive_item_ids: Sequence[int],
    rng: random.Random,
) -> dict[str, Any]:
    """Convert one current HR@3 SFT sample into a same-length preference pair."""

    user, chosen = _message_pair(sample)
    chosen_content = chosen["content"].strip()
    if chosen_content.startswith('""') and chosen_content.endswith('""'):
        chosen_content = chosen_content[1:-1]
    chosen_titles = [_clean_title(line) for line in chosen_content.splitlines() if line.strip()]
    positive_ids = [int(item_id) for item_id in positive_item_ids]
    if not chosen_titles or len(chosen_titles) != len(positive_ids):
        raise ValueError("Ranking chosen titles and positive item ids must have equal non-zero length")

    mismatched = [
        (title, item_id)
        for title, item_id in zip(chosen_titles, positive_ids, strict=True)
        if not title_index.matches(title, item_id)
    ]
    if mismatched:
        raise ValueError(
            f"Current SFT response titles do not match TSV positive IDs: {mismatched}"
        )

    candidate_ids: list[int] = []
    seen: set[int] = set()
    for title in _candidate_titles(user["content"]):
        excluded = set(positive_ids) | seen
        try:
            item_id = title_index.resolve(title, exclude=excluded)
        except ValueError:
            if any(title_index.matches(title, item_id) for item_id in excluded):
                continue
            raise
        candidate_ids.append(item_id)
        seen.add(item_id)
    if len(candidate_ids) < len(positive_ids):
        raise ValueError("Ranking prompt does not contain enough distinct negative candidates")

    negative_ids = rng.sample(candidate_ids, k=len(positive_ids))
    rejected_content = "\n".join(f'"{title_index.title(item_id)}"' for item_id in negative_ids)
    return _base_pairwise_sample(
        sample,
        user,
        chosen,
        rejected_content,
        positive_ids,
        negative_ids,
    )


def derive_auc_example(
    sample: dict[str, Any],
    title_index: TitleIndex,
    true_item_id: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Convert one current AUC Yes/No sample into a preference pair."""

    user, chosen = _message_pair(sample)
    answer = chosen["content"].strip().casefold()
    if answer not in {"yes", "no"}:
        raise ValueError(f"AUC assistant answer must be Yes or No, got {chosen['content']!r}")

    true_id = int(true_item_id)
    target_title = _target_title(user["content"])
    if answer == "yes":
        if not title_index.matches(target_title, true_id):
            raise ValueError("AUC Yes target does not match the TSV next item")
        negative_candidates = [
            title_index.resolve(title, exclude={true_id})
            for title in _negative_context_titles(user["content"])
        ]
        negative_id = rng.choice(negative_candidates)
        rejected_answer = "No"
    else:
        negative_id = title_index.resolve(target_title, exclude={true_id})
        rejected_answer = "Yes"
    return _base_pairwise_sample(
        sample,
        user,
        chosen,
        rejected_answer,
        [true_id],
        [negative_id],
    )


def _load_targets(tsv_path: str | Path, hit: int, max_sequence_length: int = 6) -> list[list[int]]:
    if hit not in {1, 3}:
        raise ValueError("hit must be 1 or 3")
    targets: list[list[int]] = []
    with Path(tsv_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 2:
                raise ValueError(f"Invalid TSV row {line_number}: expected at least two columns")
            item_ids = [int(value) for value in columns[1].split()]
            if len(item_ids) < max_sequence_length:
                continue
            if hit == 1:
                targets.append([item_ids[-1]])
            else:
                targets.append(list(reversed(item_ids[-3:])))
    return targets


def _rebase_existing_images(sample: dict[str, Any], image_root: Path) -> int:
    rebased = 0
    localized: list[str] = []
    for raw_path in sample.get("images", []):
        candidate = image_root / Path(str(raw_path)).name
        if candidate.is_file():
            localized.append(str(candidate.resolve()))
            rebased += int(str(raw_path) != str(candidate.resolve()))
        else:
            localized.append(str(raw_path))
    sample["images"] = localized
    return rebased

def _atomic_json_dump(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_preference_file(
    source_path: str | Path,
    title_path: str | Path,
    tsv_path: str | Path,
    output_path: str | Path,
    *,
    hit: int,
    seed: int = 2025,
    force: bool = False,
) -> dict[str, Any]:
    """Build a deterministic pairwise file and return a machine-readable report."""

    source = Path(source_path)
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing derived file: {output}")
    with source.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"{source} must contain a JSON list")

    targets = _load_targets(tsv_path, hit)
    if len(samples) != len(targets):
        raise ValueError(
            f"SFT/TSV sample count mismatch after filtering: {len(samples)} != {len(targets)}"
        )

    title_index = TitleIndex.from_csv(title_path)
    rng = random.Random(seed)
    derived: list[dict[str, Any]] = []
    image_root = source.parent / "images"
    rebased_images = 0
    for source_index, (sample, item_ids) in enumerate(zip(samples, targets, strict=True)):
        try:
            if hit == 1:
                result = derive_auc_example(sample, title_index, item_ids[0], rng)
            else:
                result = derive_ranking_example(sample, title_index, item_ids, rng)
        except Exception as error:
            raise ValueError(f"Failed to derive sample {source_index}: {error}") from error
        rebased_images += _rebase_existing_images(result, image_root)
        result["source_index"] = source_index
        derived.append(result)

    _atomic_json_dump(derived, output)
    return {
        "source": str(source),
        "output": str(output),
        "hit": hit,
        "seed": seed,
        "samples": len(derived),
        "duplicate_titles": len(title_index.duplicates),
        "rebased_images": rebased_images,
    }
