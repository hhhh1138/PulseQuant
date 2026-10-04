"""Dense-attention adapter used by the released Self Forcing inference path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


Tensor = torch.Tensor


@dataclass
class AdapterStats:
    calls: int = 0
    native_calls: int = 0
    padded_calls: int = 0
    rectangular_native_calls: int = 0
    fallback_calls: int = 0
    query_tokens: int = 0
    key_tokens: int = 0
    padded_query_tokens: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reason(self, name: str) -> None:
        self.reasons[name] = self.reasons.get(name, 0) + 1

    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        result["padding_amplification"] = (
            self.padded_query_tokens / self.query_tokens if self.query_tokens else 1.0
        )
        return result


class DenseAdapter:
    def __init__(self, original: Callable):
        self.original = original
        self.stats = AdapterStats()

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        self.stats.calls += 1
        self.stats.native_calls += 1
        self.stats.query_tokens += q.shape[1]
        self.stats.key_tokens += k.shape[1]
        self.stats.padded_query_tokens += q.shape[1]
        return self.original(q, k, v, **kwargs)


def build_adapter(method: str, original: Callable, **kwargs):
    if method != "dense":
        raise ValueError("This release supports dense attention only")
    return DenseAdapter(original)
