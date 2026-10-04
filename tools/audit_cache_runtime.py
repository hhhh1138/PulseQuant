#!/usr/bin/env python3
"""Print the PulseQuant core selected after importing a model runner."""

from __future__ import annotations

import argparse
import importlib
import inspect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    importlib.import_module(args.runner)
    core = importlib.import_module("spatial_single_step.int_quant_linear")
    print(f"{args.label}_CORE={core.__file__}")
    print(
        f"{args.label}_BUILD_CONTEXT="
        f"{hasattr(core, 'build_quant_cache_context')}"
    )
    print(f"{args.label}_SAVE={inspect.signature(core.save_finalized_quant_cache)}")
    print(f"{args.label}_LOAD={inspect.signature(core.load_finalized_quant_cache)}")


if __name__ == "__main__":
    main()
