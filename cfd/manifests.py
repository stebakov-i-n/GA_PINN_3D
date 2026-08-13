from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class GeometryEntry:
    case_id: str
    file_name: str
    normalized_key: str
    original_key: str
    split: str
    decision: str


def normalize_manifest_key(value: str) -> str:
    parts = [p for p in value.replace("\\", "/").split("/") if p]
    return str(PurePosixPath(*parts))


def load_manifest_entries(split_path: str | Path, selection_path: str | Path) -> tuple[list[GeometryEntry], list[str]]:
    split = _read_json(split_path)
    selection = _read_json(selection_path)
    if not isinstance(split, dict) or not isinstance(selection, dict):
        raise ManifestError("Split and selection manifests must be JSON objects.")

    case_to_split: dict[str, str] = {}
    warnings: list[str] = []
    for split_name in ("train", "val", "validation", "test"):
        for case_id in split.get(split_name, []):
            canonical_split = "validation" if split_name == "val" else split_name
            if case_id in case_to_split and case_to_split[case_id] != canonical_split:
                warnings.append(f"Contradictory split entry for {case_id}: {case_to_split[case_id]} and {canonical_split}")
            case_to_split[case_id] = canonical_split

    seen: dict[str, str] = {}
    entries: list[GeometryEntry] = []
    for original_key, decision in selection.items():
        normalized = normalize_manifest_key(original_key)
        if normalized in seen and seen[normalized] != original_key:
            warnings.append(f"Duplicate normalized selection key {normalized}: {seen[normalized]!r} and {original_key!r}")
            continue
        seen[normalized] = original_key
        parts = normalized.split("/")
        if len(parts) != 2:
            warnings.append(f"Selection key is not <case>/<file>: {original_key!r}")
            continue
        case_id, file_name = parts
        split_name = case_to_split.get(case_id)
        if split_name is None:
            warnings.append(f"Selection entry {normalized} references case absent from split manifest.")
            continue
        if decision not in {"accept", "reject"}:
            warnings.append(f"Selection entry {normalized} has unsupported decision {decision!r}.")
            continue
        if decision == "accept":
            entries.append(GeometryEntry(case_id, file_name, normalized, original_key, split_name, decision))
    entries.sort(key=lambda e: (e.split, e.case_id, e.file_name))
    return entries, warnings


def summarize_entries(entries: list[GeometryEntry]) -> dict[str, Any]:
    by_split = {"train": 0, "validation": 0, "test": 0}
    for entry in entries:
        by_split[entry.split] = by_split.get(entry.split, 0) + 1
    return {"accepted_count": len(entries), "by_split": by_split}


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {path}: {exc}") from exc
