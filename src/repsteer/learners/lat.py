"""Linear Artificial Tomography over paired contrastive differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import Tensor

from repsteer.capture import ActivationStore
from repsteer.data import ContrastivePairs

from .base import (
    ContrastiveLearner,
    activation_matrix,
    artifact_metadata,
    canonicalize_component_signs,
    normalize_vector,
    weighted_mean,
)


def lat_components(
    positive: Tensor,
    negative: Tensor,
    *,
    weights: Tensor | None = None,
    n_components: int = 1,
    center: bool = True,
    normalize: str | None = "l2",
) -> tuple[Tensor, Tensor, Tensor]:
    """Principal components of paired differences, sign-aligned to their mean."""

    if positive.ndim != 2 or negative.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("LAT requires equally shaped positive/negative matrices")
    differences = positive - negative
    samples, hidden = differences.shape
    if not 1 <= n_components <= min(samples, hidden):
        raise ValueError(
            f"n_components must be in [1, {min(samples, hidden)}], got {n_components}"
        )
    if weights is None:
        weights = torch.ones(
            samples, dtype=differences.dtype, device=differences.device
        )
    else:
        weights = torch.as_tensor(
            weights, dtype=differences.dtype, device=differences.device
        ).flatten()
    mean_difference = weighted_mean(differences, weights)
    matrix = differences - mean_difference if center else differences
    weighted = matrix * weights.clamp_min(0).sqrt()[:, None]
    _, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    basis = vh[:n_components].clone()

    # Centered identical differences contain no variance.  Their only meaningful
    # steering direction is the contrastive mean itself.
    if float(torch.linalg.vector_norm(weighted)) <= torch.finfo(weighted.dtype).eps:
        first = normalize_vector(mean_difference.clone(), normalize)
        basis[0] = first
        if n_components > 1:
            basis[1:] = canonicalize_component_signs(basis[1:])
    else:
        for index in range(n_components):
            alignment = torch.dot(basis[index], mean_difference)
            if float(alignment) < 0:
                basis[index] = -basis[index]
            elif float(alignment) == 0:
                basis[index : index + 1] = canonicalize_component_signs(
                    basis[index : index + 1]
                )
            basis[index] = normalize_vector(basis[index], normalize)
    denominator = max(float(weights.sum()) - 1.0, 1.0)
    explained_variance = singular_values[:n_components].square() / denominator
    return basis, mean_difference, explained_variance


def compute_lat(
    positive: Tensor,
    negative: Tensor,
    *,
    weights: Tensor | None = None,
    center: bool = True,
    normalize: str | None = "l2",
) -> Tensor:
    return lat_components(
        positive,
        negative,
        weights=weights,
        n_components=1,
        center=center,
        normalize=normalize,
    )[0][0]


@dataclass
class LAT(ContrastiveLearner):
    n_components: int = 1
    center: bool = True
    method: ClassVar[str] = "lat"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_components = int(self.n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

    def fit(
        self,
        model: Any,
        data: ContrastivePairs,
        *,
        store: ActivationStore | None = None,
    ) -> Any:
        from repsteer.artifacts import DirectionArtifact, SubspaceArtifact

        if not data.paired:
            raise ValueError("LAT requires paired positive/negative records")
        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        pair_weights = torch.as_tensor(
            data.pair_weights, dtype=positive.dtype, device=positive.device
        )
        basis, mean, explained_variance = lat_components(
            positive,
            negative,
            weights=pair_weights,
            n_components=self.n_components,
            center=self.center,
            normalize=self.normalize,
        )
        basis = basis.detach().to(device="cpu", dtype=torch.float32)
        mean = mean.detach().to(device="cpu", dtype=torch.float32)
        explained_variance = explained_variance.detach().to(
            device="cpu", dtype=torch.float32
        )
        artifact_type = "direction" if self.n_components == 1 else "subspace"
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type=artifact_type,
            hidden_size=basis.shape[1],
            extra_config={
                "n_components": self.n_components,
                "center": self.center,
                "pairs": positive.shape[0],
                "sign_alignment": "weighted_mean_difference",
            },
        )
        if self.n_components == 1:
            return DirectionArtifact(metadata=metadata, direction=basis[0])
        return SubspaceArtifact(
            metadata=metadata,
            basis=basis,
            mean=mean,
            explained_variance=explained_variance,
        )


__all__ = ["LAT", "compute_lat", "lat_components"]
