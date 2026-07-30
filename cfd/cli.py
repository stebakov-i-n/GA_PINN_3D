from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .manifests import ManifestError, load_manifest_entries, summarize_entries


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cfd", description="Independent CFD preprocessing/reference tools.")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("list-geometries", help="List accepted geometries from read-only manifests.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_list_geometries)

    p = sub.add_parser("validate-geometry", help="Validate one vascular STL bundle and write a run report.")
    p.add_argument("--config", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--direction", choices=["forward", "reverse"], required=True)
    p.add_argument("--output-root", required=True)
    p.set_defaults(func=cmd_validate_geometry)

    p = sub.add_parser("cylinder", help="Generate analytical cylinder reference and OpenFOAM case.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--run-openfoam", action="store_true")
    p.set_defaults(func=cmd_cylinder)

    p = sub.add_parser("vessel", help="Validate or generate a vascular OpenFOAM meshing case.")
    p.add_argument("--config", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--direction", choices=["forward", "reverse"], required=True)
    p.add_argument("--output-root", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--generate-openfoam", action="store_true")
    mode.add_argument("--mesh-openfoam", action="store_true")
    mode.add_argument("--run-openfoam", action="store_true")
    p.set_defaults(func=cmd_vessel)

    p = sub.add_parser("patch-cylinder-inlet", help="Patch a meshed cylinder case with a parabolic inlet profile.")
    p.add_argument("--config", required=True)
    p.add_argument("--case-dir", required=True)
    p.set_defaults(func=cmd_patch_cylinder_inlet)
    return parser


def cmd_list_geometries(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    split = Path(config["geometry"]["split_manifest"])
    selection = Path(config["geometry"]["selection_manifest"])
    entries, warnings = load_manifest_entries(split, selection)
    summary = summarize_entries(entries)
    print(f"accepted_count: {summary['accepted_count']}")
    for split_name, count in summary["by_split"].items():
        print(f"{split_name}: {count}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    for entry in entries:
        print(f"{entry.split}\t{entry.normalized_key}\t(original={entry.original_key})")
    dataset_root = Path(config["geometry"]["dataset_root"])
    if not dataset_root.exists():
        print(f"dataset_root_missing: {dataset_root}")
    return 0


def cmd_validate_geometry(args: argparse.Namespace) -> int:
    from .geometry import validate_bundle
    from .reporting import create_run_dir, finish_run, start_metadata

    config = load_config(args.config)
    run_dir = create_run_dir(args.output_root, args.geometry.replace("/", "_").replace(".stl", ""), args.direction, config)
    metadata = start_metadata(config, sys.argv if sys.argv else ["python", "-m", "cfd"], args.geometry, args.direction)
    metrics = {}
    try:
        result = validate_bundle(config["geometry"]["dataset_root"], args.geometry, args.direction, config, run_dir / "figures")
        metrics["geometry_validation"] = result
        status = "failed" if result["failures"] else "generated_not_executed"
        failure = "; ".join(result["failures"]) if result["failures"] else None
    except Exception as exc:
        status = "failed"
        failure = str(exc)
        metrics["failure"] = failure
    finish_run(run_dir, config, metadata, metrics, status, failure)
    print(run_dir)
    return 1 if status == "failed" else 0


def cmd_cylinder(args: argparse.Namespace) -> int:
    from .cylinder import run_cylinder

    config = load_config(args.config)
    run_dir = run_cylinder(config, args.output_root, sys.argv if sys.argv else ["python", "-m", "cfd"], args.run_openfoam)
    print(run_dir)
    return 0


def cmd_vessel(args: argparse.Namespace) -> int:
    from .vessel import run_vessel

    config = load_config(args.config)
    if args.validate_only:
        mode = "validate_only"
    elif args.generate_openfoam:
        mode = "generate_openfoam"
    elif args.mesh_openfoam:
        mode = "mesh_openfoam"
    else:
        mode = "run_openfoam"
    run_dir = run_vessel(config, args.geometry, args.direction, args.output_root, sys.argv if sys.argv else ["python", "-m", "cfd"], mode)
    print(run_dir)
    return 0


def cmd_patch_cylinder_inlet(args: argparse.Namespace) -> int:
    from .openfoam import patch_cylinder_inlet_from_mesh

    config = load_config(args.config)
    count = patch_cylinder_inlet_from_mesh(args.case_dir, config)
    print(f"patched_inlet_faces: {count}")
    return 0
