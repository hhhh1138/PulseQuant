"""Prompt-source helpers for separating calibration from evaluation."""

from pathlib import Path


def load_nonempty_prompts(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def select_calibration_prompts(
    prompts: list[str], indices: list[int], expected_count: int
) -> list[str]:
    if len(indices) != expected_count:
        raise ValueError("calibration prompt index count mismatch")
    if not indices or min(indices) < 0 or max(indices) >= len(prompts):
        raise ValueError("calibration prompt index is outside the calibration prompt file")
    return [prompts[index] for index in indices]
