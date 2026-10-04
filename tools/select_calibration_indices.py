#!/usr/bin/env python3
"""Resolve calibration indices against a user-supplied, non-empty prompt file."""

import argparse
from pathlib import Path


def select_indices(size: int, count: str = "auto", indices: str = "") -> list[int]:
    if size < 1:
        raise ValueError("calibration prompt file is empty")
    if indices.strip():
        selected = [int(item.strip()) for item in indices.split(",")]
        if len(set(selected)) != len(selected):
            raise ValueError("calibration indices must be unique")
        if any(index < 0 or index >= size for index in selected):
            raise ValueError(f"calibration indices must be between 0 and {size - 1}")
        if count != "auto" and int(count) != len(selected):
            raise ValueError("calibration count must match explicit indices")
        return selected
    total = min(10, size) if count == "auto" else int(count)
    if not 1 <= total <= size:
        raise ValueError(f"calibration count must be between 1 and {size}")
    return [0] if total == 1 else [i * (size - 1) // (total - 1) for i in range(total)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--count", default="auto")
    parser.add_argument("--indices", default="")
    args = parser.parse_args()
    prompts = [line for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        selected = select_indices(len(prompts), args.count, args.indices)
    except ValueError as exc:
        parser.error(str(exc))
    print(len(selected), ",".join(map(str, selected)))


if __name__ == "__main__":
    main()
