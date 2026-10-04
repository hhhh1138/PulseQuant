#!/usr/bin/env python3
"""Combine disjoint generation shards using the supplied manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.samefile(source):
            return
        raise RuntimeError(f"refusing to overwrite {destination}")
    os.link(source, destination)


def collect_records(config_dir: Path, manifest: list, shards: list[Path] | None = None):
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest must be a non-empty JSON list")
    shard_dirs = shards if shards is not None else sorted(config_dir.glob("shard*"))
    if not shard_dirs:
        raise ValueError("no shards found; supply --shard directories")
    records = {}
    sources = []
    combined = (config_dir / "combined").resolve()
    for shard in shard_dirs:
        if shard.resolve() == combined:
            raise ValueError("combined output cannot be used as an input shard")
        video_dir = shard / "videos"
        if not video_dir.is_dir():
            raise ValueError(f"missing shard video directory: {video_dir}")
        for video in sorted(video_dir.glob("*.mp4")):
            prefix = video.stem.split("_", 1)[0]
            if not prefix.isdigit():
                raise ValueError(f"video filename must begin with a numeric index: {video}")
            index = int(prefix)
            if index in records:
                raise RuntimeError(f"duplicate global index {index}")
            metadata = video.with_suffix(".json")
            if not metadata.is_file() or video.stat().st_size == 0:
                raise RuntimeError(f"incomplete output {video}")
            records[index] = json.loads(metadata.read_text(encoding="utf-8"))
            sources.extend((video, metadata))
            audio = video.with_suffix(".audio.pt")
            if audio.is_file():
                sources.append(audio)
    expected = set(range(len(manifest)))
    if set(records) != expected:
        missing = sorted(expected - records.keys())
        extra = sorted(records.keys() - expected)
        raise RuntimeError(f"global-index audit failed: missing={missing}, extra={extra}")
    return records, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append",
                        help="Input shard directory; repeat as needed. Default: config-dir/shard*.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records, sources = collect_records(args.config_dir, manifest, args.shard)
    combined = args.config_dir / "combined"
    videos = combined / "videos"
    # Check destination conflicts before creating any links.
    for source in sources:
        destination = videos / source.name
        if destination.is_symlink() or (destination.exists() and not destination.samefile(source)):
            raise RuntimeError(f"refusing to overwrite {destination}")
    videos.mkdir(parents=True, exist_ok=True)
    for source in sources:
        link(source, videos / source.name)
    (combined / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (combined / "generation_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(manifest)):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    (combined / "ASSEMBLED").write_text(f"{len(records)} unique videos\n", encoding="utf-8")
    print(json.dumps({"videos": len(records), "config": args.config_dir.name}))


if __name__ == "__main__":
    main()
