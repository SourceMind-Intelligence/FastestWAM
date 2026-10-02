#!/usr/bin/env python3
"""Fetch pinned public LIBERO-Plus artifacts and verify their published hashes."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


SOURCES = {
    "dataset": ("Sylvest/libero_plus_rlds", "dataset", "libero_plus_rlds_blobs.json"),
    "assets": ("Sylvest/LIBERO-plus", "dataset", "assets_blobs.json"),
    "start": (
        "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "model",
        "start_blobs.json",
    ),
    "result": (
        "Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata",
        "model",
        "result_blobs.json",
    ),
}


def selected_files(group, siblings):
    names = [entry["rfilename"] for entry in siblings]
    if group == "dataset":
        return [name for name in names if name == "README.md" or name.startswith("libero_plus_mixdata.")]
    if group == "assets":
        return [name for name in names if name in {"README.md", "assets.zip"}]
    return [name for name in names if not name.startswith(".gitattributes")]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path, help="Artifact directory")
    parser.add_argument("--group", choices=[*SOURCES, "all"], required=True)
    parser.add_argument("--list", action="store_true", help="Show pinned files without downloading")
    parser.add_argument("--max-bytes", type=int, help="Skip files larger than this size")
    args = parser.parse_args()
    groups = SOURCES if args.group == "all" else {args.group: SOURCES[args.group]}
    metadata_dir = Path(__file__).resolve().parent / "metadata"

    for group, (repo_id, repo_type, metadata_file) in groups.items():
        info = json.loads((metadata_dir / metadata_file).read_text())
        revision = info["sha"]
        siblings = {entry["rfilename"]: entry for entry in info["siblings"]}
        output = args.dest / group
        for name in selected_files(group, info["siblings"]):
            entry = siblings[name]
            expected = entry.get("lfs", {}).get("sha256")
            size = entry.get("size")
            if args.max_bytes is not None and size > args.max_bytes:
                continue
            print(f"{group}: {name} ({size} bytes) @ {revision}", flush=True)
            if args.list:
                continue
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=name,
                    revision=revision,
                    local_dir=output,
                )
            )
            if path.stat().st_size != size:
                raise RuntimeError(f"size mismatch: {path}")
            if expected and sha256(path) != expected:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
            print(f"verified: {path}", flush=True)


if __name__ == "__main__":
    main()
