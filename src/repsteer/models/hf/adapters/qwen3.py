from __future__ import annotations

from .base import DecoderOnlyAdapter


class Qwen3Adapter(DecoderOnlyAdapter):
    architecture_name = "qwen3"
    model_types = frozenset({"qwen3", "qwen3_moe"})
    class_prefixes = ("Qwen3",)
