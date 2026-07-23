"""Activation transformation operators."""

from .add import Add, Subtract
from .base import Operator, operator_from_dict
from .projection import RemoveProjection
from .replace import Replace
from .sae import Ablate, Clamp, LatentAblate, LatentClamp, SAEAblate, SAEClamp

__all__ = [
    "Add",
    "Ablate",
    "Clamp",
    "LatentAblate",
    "LatentClamp",
    "Operator",
    "RemoveProjection",
    "Replace",
    "SAEAblate",
    "SAEClamp",
    "Subtract",
    "operator_from_dict",
]
