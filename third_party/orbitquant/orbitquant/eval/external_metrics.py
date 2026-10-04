# Modified for PulseQuant: external video-benchmark integration removed.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def summarize_geneval_results(results_jsonl: str | Path, output_json: str | Path) -> dict[str, Any]:
    path = Path(results_jsonl)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"GenEval results file is empty: {path}")
    per_tag: dict[str, dict[str, int]] = {}
    correct_count = 0
    prompt_correct: dict[str, bool] = {}
    for record in records:
        tag = str(record.get("tag", "unknown"))
        correct = bool(record.get("correct", False))
        if record.get("metadata") is not None:
            metadata_key = str(record["metadata"])
            prompt_correct[metadata_key] = prompt_correct.get(metadata_key, False) or correct
        bucket = per_tag.setdefault(tag, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if correct:
            bucket["correct"] += 1
            correct_count += 1
    tags = {
        tag: {
            "score": values["correct"] / values["total"],
            "correct": values["correct"],
            "total": values["total"],
        }
        for tag, values in sorted(per_tag.items())
    }
    per_task = {tag: values["score"] for tag, values in tags.items()}
    summary = {
        "overall": sum(per_task.values()) / len(per_task),
        "image_accuracy": correct_count / len(records),
        "records": len(records),
        "per_task": per_task,
        "tags": tags,
    }
    if prompt_correct:
        summary["prompt_accuracy"] = sum(prompt_correct.values()) / len(prompt_correct)
        summary["prompts"] = len(prompt_correct)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
