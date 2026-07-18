from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def stable_order(interventions: Iterable[Any]) -> tuple[Any, ...]:
    """Return priority/declaration order without reordering operators by kind."""

    return tuple(
        value
        for _, value in sorted(
            enumerate(interventions),
            key=lambda pair: (int(getattr(pair[1], "priority", 0)), pair[0]),
        )
    )


def is_additive(operator: Any) -> bool:
    return type(operator).__name__.lower() in {"add", "subtract"}


__all__ = ["is_additive", "stable_order"]
