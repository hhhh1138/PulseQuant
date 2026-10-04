#!/usr/bin/env python3
"""Decode-level integrity audit for an arbitrary directory of generated videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, help="Optional expected number of MP4 files.")
    parser.add_argument("--width", type=int, help="Optional required width.")
    parser.add_argument("--height", type=int, help="Optional required height.")
    parser.add_argument("--min-frames", type=int, default=1)
    args = parser.parse_args()
    if args.min_frames < 1 or any(value is not None and value < 1
                                 for value in (args.expected_count, args.width, args.height)):
        parser.error("count, dimensions and min-frames must be positive")
    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"no MP4 files found in {args.videos}")
    if args.expected_count is not None and len(videos) != args.expected_count:
        raise RuntimeError(f"expected {args.expected_count} videos, found {len(videos)}")
    rows = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot decode {video}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            decoded = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
        finally:
            capture.release()
        if (width <= 0 or height <= 0 or decoded < args.min_frames
                or (declared > 0 and declared != decoded)
                or (args.width is not None and width != args.width)
                or (args.height is not None and height != args.height)):
            raise RuntimeError(
                f"invalid video {video}: {width}x{height}, declared={declared}, decoded={decoded}")
        rows.append({"file": video.name, "bytes": video.stat().st_size, "width": width,
                     "height": height, "declared_frames": declared, "decoded_frames": decoded})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"count": len(rows), "videos": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"decoded": len(rows)}))


if __name__ == "__main__":
    main()
