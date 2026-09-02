"""Re-upload main.py training artifacts to ClearML for a run whose
upload_artifact calls didn't make it to remote storage.

main.py always saves its outputs locally to trained_models/<timestamp>/
before uploading them to ClearML, so recovery just means pointing back at
that local run folder and re-running the same upload_artifact calls.
"""

import argparse
import os

from clearml import Task

ARTIFACT_FILES = {
    "model": "model.pth",
    "model_best": "model_best.pth",
    "history": "history.json",
    "history_val": "history_val.json",
    "optimizer_all": "optimizer_all.pth",
    "optimizer_phi": "optimizer_phi.pth",
    "optimizer_flow": "optimizer_flow.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default='19e30699b1244fb89166af2e1e7bf5e1', help="ClearML task id whose artifacts failed to upload.")
    parser.add_argument("--run-dir", default='trained_models/2026-08-25_19-38-00', help="Local trained_models/<timestamp> folder for that run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task = Task.init(auto_connect_frameworks=False)

    found = {
        name: path
        for name, filename in ARTIFACT_FILES.items()
        if os.path.exists(path := os.path.join(args.run_dir, filename))
    }
    if not found:
        print(f"No known artifact files found in {args.run_dir}")
        return

    for name, path in found.items():
        print(f"Uploading {name} <- {path}")
        task.upload_artifact(name, artifact_object=path)

    for name in found:
        artifact = task.artifacts.get(name)
        local_copy = artifact.get_local_copy() if artifact else None
        status = "OK" if local_copy and os.path.exists(local_copy) else "FAILED"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
