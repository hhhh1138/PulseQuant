"""Shard-invariant prompt seed resolution."""


def resolve_prompt_seed(
    base_seed: int,
    seed_mode: str,
    seed_index_origin: str,
    prompt_offset: int,
    local_index: int,
) -> int:
    if seed_mode == "fixed":
        return base_seed
    if seed_mode != "per_prompt":
        raise ValueError(f"unknown seed mode: {seed_mode}")
    if seed_index_origin == "local":
        return base_seed + local_index
    if seed_index_origin == "absolute":
        return base_seed + prompt_offset + local_index
    raise ValueError(f"unknown seed index origin: {seed_index_origin}")
