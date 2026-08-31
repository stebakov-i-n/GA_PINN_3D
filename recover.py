"""Recover local training outputs left on a ClearML agent.

This script is intended to be launched as a small ClearML task on the same
queue/agent family that ran a training task whose artifacts failed to upload.
It searches common ClearML agent directories for PEFT adapter files and uploads
the recovered model directory as this recovery task's ``final_model_dir``
artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from clearml import Task


MODEL_MARKERS = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
}
LOG_MARKERS = {
    "trainer_log_history.json",
    "trainer_state.json",
    "train_metrics.json",
    "resampling_report.json",
}
OUTPUT_MARKERS = {
    "predictions.jsonl",
    "predictions.metadata.json",
    "metrics_summary.csv",
    "metrics_summary.json",
    "metrics_all_samples.csv",
    "metrics_known_only.csv",
    "per_class_metrics.csv",
    "confusion_matrices_all_samples.csv",
    "metrics_by_dataset.csv",
    "parsed_predictions.csv",
}


def parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-task-id", "--source_task_id", required=True)
    parser.add_argument(
        "--search-roots",
        "--search_roots",
        default="/root/.clearml,/clearml_agent_cache,/tmp",
        help="Comma-separated roots to scan on the ClearML agent.",
    )
    parser.add_argument(
        "--expected-dir-name",
        "--expected_dir_name",
        default="medgemma_v2_resampling_field_value_loss_aug_6epochs_fold0",
        help="Prefer recovered directories with this name in their path.",
    )
    parser.add_argument(
        "--max-files",
        "--max_files",
        type=int,
        default=1_500_000,
        help="Safety limit for scanned files.",
    )
    parser.add_argument(
        "--require-expected-dir-name",
        "--require_expected_dir_name",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Ignore adapter candidates whose path does not contain --expected-dir-name.",
    )
    parser.add_argument(
        "--no-upload",
        "--no_upload",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Only print recovered candidates; do not upload artifacts to ClearML storage.",
    )
    parser.add_argument(
        "--recovery-kind",
        "--recovery_kind",
        choices=["auto", "model", "inference_outputs"],
        default="auto",
        help="Recover a model adapter directory or an inference output directory.",
    )
    return parser.parse_args()


def iter_files(root: Path, max_files: int):
    scanned = 0
    for current_root, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__", "node_modules"}
        ]
        for filename in filenames:
            scanned += 1
            if scanned > max_files:
                raise RuntimeError(f"Scanned more than {max_files} files under {root}")
            yield Path(current_root) / filename


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def score_model_dir(path: Path, expected_dir_name: str) -> tuple[int, str]:
    files = {child.name for child in path.iterdir() if child.is_file()}
    score = 0
    if "adapter_config.json" in files:
        score += 10
    if "adapter_model.safetensors" in files:
        score += 10
    if "adapter_model.bin" in files:
        score += 8
    if "train_metrics.json" in files:
        score += 3
    if "trainer_log_history.json" in files:
        score += 3
    if expected_dir_name and expected_dir_name in str(path):
        score += 5
    if path.name.startswith("checkpoint-"):
        score -= 1
    return score, str(path)


def build_model_archive(model_dir: Path, source_task_id: str) -> Path:
    archive_base = model_dir.parent / f"{model_dir.name}_{source_task_id}_recovered_final_model_dir"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=model_dir))
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create non-empty recovered model archive: {archive_path}")
    return archive_path


def build_outputs_archive(output_dir: Path, source_task_id: str) -> Path:
    archive_base = output_dir.parent / f"{output_dir.name}_{source_task_id}_recovered_outputs"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_dir))
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create non-empty recovered outputs archive: {archive_path}")
    return archive_path


def upload_required_artifact(task: Task, name: str, path: Path | dict[str, Any]) -> None:
    uploaded = task.upload_artifact(name, str(path) if isinstance(path, Path) else path)
    if uploaded is False:
        raise RuntimeError(f"ClearML upload_artifact returned False for {name}: {path}")

    task.flush(wait_for_uploads=True)
    artifact = task.artifacts.get(name)
    if artifact is None:
        raise RuntimeError(f"ClearML artifact metadata was not created for {name}")

    local_copy = artifact.get_local_copy()
    if not local_copy or not Path(local_copy).exists():
        raise RuntimeError(
            f"ClearML artifact {name} was registered but is not downloadable. "
            f"URL: {getattr(artifact, 'url', None)}"
        )


def artifact_safe_name(path: Path, selected: Path) -> str:
    try:
        relative = path.relative_to(selected)
    except ValueError:
        relative = Path(path.name)
    safe = "_".join(relative.parts)
    safe = safe.replace(".", "_")
    return safe


def score_output_dir(path: Path, expected_dir_name: str) -> tuple[int, str]:
    files = {child.name for child in path.iterdir() if child.is_file()}
    eval_dir = path / "evaluation"
    eval_files = {child.name for child in eval_dir.iterdir() if eval_dir.is_dir() and child.is_file()}
    score = 0
    if "predictions.jsonl" in files:
        score += 10
    if "predictions.metadata.json" in files:
        score += 5
    if "metrics_summary.json" in eval_files:
        score += 5
    if "metrics_known_only.csv" in eval_files:
        score += 4
    if "confusion_matrices_all_samples.csv" in eval_files:
        score += 4
    if "per_class_metrics.csv" in eval_files:
        score += 4
    if expected_dir_name and expected_dir_name in str(path):
        score += 5
    return score, str(path)


def output_candidate_root(path: Path) -> Path:
    if path.parent.name == "evaluation":
        return path.parent.parent
    return path.parent


def main() -> None:
    args = parse_args()

    task = Task.init(
        project_name="pershin-medailab/VLM_Embryo",
        task_name=f"recover outputs for {args.source_task_id}",
        task_type=Task.TaskTypes.data_processing,
    )
    task.connect(vars(args))

    roots = [Path(item.strip()) for item in args.search_roots.split(",") if item.strip()]
    found_files: dict[str, list[str]] = {name: [] for name in sorted(MODEL_MARKERS | LOG_MARKERS | OUTPUT_MARKERS)}

    for root in roots:
        if not root.exists():
            print(f"SKIP missing root: {root}")
            continue
        print(f"SCAN root: {root}")
        try:
            for path in iter_files(root, args.max_files):
                if path.name in found_files:
                    found_files[path.name].append(str(path))
        except PermissionError as exc:
            print(f"SKIP permission error under {root}: {exc}")
        except OSError as exc:
            print(f"SKIP os error under {root}: {exc}")

    candidate_dirs = set()
    for marker in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        for item in found_files.get(marker, []):
            candidate_dirs.add(Path(item).parent)

    valid_model_dirs = []
    for candidate in candidate_dirs:
        try:
            names = {child.name for child in candidate.iterdir() if child.is_file()}
        except OSError:
            continue
        if "adapter_config.json" in names and (
            "adapter_model.safetensors" in names or "adapter_model.bin" in names
        ):
            valid_model_dirs.append(candidate)

    if args.require_expected_dir_name:
        valid_model_dirs = [
            path for path in valid_model_dirs if args.expected_dir_name in str(path)
        ]

    valid_model_dirs.sort(
        key=lambda item: score_model_dir(item, args.expected_dir_name),
        reverse=True,
    )

    output_candidate_dirs = set()
    for marker in OUTPUT_MARKERS:
        for item in found_files.get(marker, []):
            output_candidate_dirs.add(output_candidate_root(Path(item)))

    valid_output_dirs = []
    for candidate in output_candidate_dirs:
        if args.require_expected_dir_name and args.expected_dir_name not in str(candidate):
            continue
        if (candidate / "predictions.jsonl").exists() or (candidate / "evaluation" / "metrics_summary.json").exists():
            valid_output_dirs.append(candidate)

    valid_output_dirs.sort(
        key=lambda item: score_output_dir(item, args.expected_dir_name),
        reverse=True,
    )

    manifest: dict[str, Any] = {
        "source_task_id": args.source_task_id,
        "search_roots": [str(root) for root in roots],
        "found_files": found_files,
        "valid_model_dirs": [str(path) for path in valid_model_dirs],
        "valid_output_dirs": [str(path) for path in valid_output_dirs],
        "selected_model_dir": str(valid_model_dirs[0]) if valid_model_dirs else None,
        "selected_output_dir": str(valid_output_dirs[0]) if valid_output_dirs else None,
    }

    for path in valid_model_dirs[:5]:
        print("MODEL_DIR_CANDIDATE:", path)
        for name in sorted(MODEL_MARKERS | LOG_MARKERS):
            file_path = path / name
            if file_path.exists():
                print(f"  {name}: {file_size(file_path)} bytes")

    for path in valid_output_dirs[:5]:
        print("OUTPUT_DIR_CANDIDATE:", path)
        for name in sorted(OUTPUT_MARKERS):
            candidates = [path / name, path / "evaluation" / name]
            for file_path in candidates:
                if file_path.exists():
                    print(f"  {file_path.relative_to(path)}: {file_size(file_path)} bytes")

    if args.no_upload:
        print("NO_UPLOAD_MODE: not uploading recovery_manifest or recovered artifacts")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        upload_required_artifact(task, "recovery_manifest", manifest)

    if args.recovery_kind in {"auto", "inference_outputs"} and valid_output_dirs:
        selected_output = valid_output_dirs[0]
        print("SELECTED_OUTPUT_DIR:", selected_output)
        archive_path = build_outputs_archive(selected_output, args.source_task_id)
        print("OUTPUTS_ARCHIVE:", archive_path, file_size(archive_path), "bytes")
        if args.no_upload:
            print("NO_UPLOAD_MODE: not uploading recovered_outputs")
            return

        upload_required_artifact(task, "recovered_outputs", archive_path)
        for path in sorted(selected_output.rglob("*")):
            if path.is_file() and path.name in OUTPUT_MARKERS:
                artifact_name = "output_" + artifact_safe_name(path, selected_output)
                upload_required_artifact(task, artifact_name, path)
        print("RECOVERY_TASK_ID:", task.id)
        print("RECOVERED_OUTPUTS_ARTIFACT: recovered_outputs")
        if args.recovery_kind == "inference_outputs":
            return

    selected = valid_model_dirs[0] if valid_model_dirs else None
    if selected is None:
        print("NO_VALID_MODEL_DIR_FOUND")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    print("SELECTED_MODEL_DIR:", selected)
    archive_path = build_model_archive(selected, args.source_task_id)
    print("MODEL_ARCHIVE:", archive_path, file_size(archive_path), "bytes")
    if args.no_upload:
        print("NO_UPLOAD_MODE: not uploading final_model_dir")
        return

    upload_required_artifact(task, "final_model_dir", archive_path)

    for log_name in sorted(LOG_MARKERS):
        log_path = selected / log_name
        if log_path.exists():
            upload_required_artifact(task, log_name.removesuffix(".json"), log_path)

    trainer_states = sorted(
        {Path(item) for item in found_files.get("trainer_state.json", []) if selected in Path(item).parents or Path(item).parent == selected}
    )
    for state_path in trainer_states:
        artifact_name = "trainer_state_" + artifact_safe_name(state_path, selected)
        upload_required_artifact(task, artifact_name, state_path)

    print("RECOVERY_TASK_ID:", task.id)
    print("RECOVERED_FINAL_MODEL_DIR_ARTIFACT: final_model_dir")


if __name__ == "__main__":
    main()
