#!/usr/bin/env python3
"""Read-only syntax and heuristic privacy checks for a source release."""
from __future__ import annotations
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "cloud key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
        "HF token": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
        "private path": re.compile(r"/(?:mnt|home|Users)/[^\s\"']+|[A-Za-z]:\\\\?Users\\"),
        "private endpoint": re.compile(r"https?://(?:localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)(?=[:/\s])"),
        "assigned credential": re.compile(r'''(?i)(?:api_key|access_key_id|secret_access_key|password)\s*[=:]\s*["'][^"'\s]{8,}["']'''),
    }
    errors = []
    checked = 0
    bash = shutil.which("bash")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(p in {".git", "__pycache__", "runs", ".venv"} for p in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root)
        if rel.parts[:2] == ("assets", "figures") and path.suffix == ".png":
            with path.open("rb") as stream:
                if stream.read(8) != bytes([137, 80, 78, 71, 13, 10, 26, 10]):
                    errors.append(f"Invalid figure PNG header: {rel}")
            continue
        if rel.parts[:2] == ("assets", "gallery") and path.suffix == ".gif":
            with path.open("rb") as stream:
                if stream.read(6) not in {b"GIF87a", b"GIF89a"}:
                    errors.append(f"Invalid gallery GIF header: {rel}")
            continue
        if path.suffix not in {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".cpp", ".h", ".cu", ".cuh", ".c", ".hpp", ".jinja", ".pyi"} and path.name not in {".env.example", ".gitignore", ".gitattributes", "LICENSE", "NOTICE", "py.typed", "CMakeLists.txt"}:
            errors.append(f"Review unexpected release file: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        checked += 1
        try:
            if path.suffix == ".py":
                ast.parse(text, filename=str(rel))
            elif path.suffix == ".json":
                json.loads(text)
            elif path.suffix == ".sh" and bash:
                result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(f"Bash syntax failed: {rel}")
        except (SyntaxError, ValueError) as exc:
            errors.append(f"Parse error: {rel}: {exc}")
        if path == Path(__file__).resolve():
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns.items():
                # Upstream Diffusers uses this generic cloud mount convention.
                checked_line = line.replace("/mnt/buckets/", "BUCKET_MOUNT/")
                if pattern.search(checked_line):
                    errors.append(f"{rel}:{number}: possible {label}")
    for error in errors:
        print(error)  # Do not echo the potentially sensitive value.
    print(f"Checked {checked} text files; findings: {len(errors)}")
    if not bash:
        print("Bash unavailable: shell syntax checks skipped.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
