"""Centered SVD principal-component learner."""

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
    sample_weights,
    weighted_mean,
)


def principal_components(
    activations: Tensor,
    *,
    n_components: int = 1,
    weights: Tensor | None = None,
    center: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(basis, mean, explained_variance)`` using deterministic SVD signs."""

    if activations.ndim != 2:
        raise ValueError("PCA activations must be [samples, hidden]")
    samples, hidden = activations.shape
    if not 1 <= n_components <= min(samples, hidden):
        raise ValueError(
            f"n_components must be in [1, {min(samples, hidden)}], got {n_components}"
        )
    if weights is None:
        weights = torch.ones(
            samples, dtype=activations.dtype, device=activations.device
        )
    else:
        weights = torch.as_tensor(
            weights, dtype=activations.dtype, device=activations.device
        ).flatten()
    mean = (
        weighted_mean(activations, weights)
        if center
        else torch.zeros(hidden, dtype=activations.dtype, device=activations.device)
    )
    centered = activations - mean if center else activations
    weighted = centered * weights.clamp_min(0).sqrt()[:, None]
    # full_matrices=False avoids constructing a hidden_size squared matrix.
    _, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    basis = canonicalize_component_signs(vh[:n_components])
    denominator = max(float(weights.sum()) - 1.0, 1.0)
    explained_variance = singular_values[:n_components].square() / denominator
    return basis, mean, explained_variance


compute_pca = principal_components


@dataclass
class PCA(ContrastiveLearner):
    n_components: int = 1
    center: bool = True
    normalize: str | None = "orthonormal"
    method: ClassVar[str] = "pca"

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
        from repsteer.artifacts import SubspaceArtifact

        positive_batch, negative_batch = self._capture_pair(model, data, store)
        positive = activation_matrix(positive_batch, name="positive activations")
        negative = activation_matrix(negative_batch, name="negative activations")
        activations = torch.cat((positive, negative), dim=0)
        weights = torch.cat(
            (
                sample_weights(positive_batch, positive),
                sample_weights(negative_batch, negative),
            ),
            dim=0,
        )
        basis, mean, explained_variance = principal_components(
            activations,
            n_components=self.n_components,
            weights=weights,
            center=self.center,
        )
        basis = basis.detach().to(device="cpu", dtype=torch.float32)
        mean = mean.detach().to(device="cpu", dtype=torch.float32)
        explained_variance = explained_variance.detach().to(
            device="cpu", dtype=torch.float32
        )
        metadata = artifact_metadata(
            learner=self,
            model=model,
            data=data,
            artifact_type="subspace",
            hidden_size=basis.shape[1],
            extra_config={
                "n_components": self.n_components,
                "center": self.center,
                "samples": activations.shape[0],
                "svd": "torch.linalg.svd",
            },
        )
        return SubspaceArtifact(
            metadata=metadata,
            basis=basis,
            mean=mean,
            explained_variance=explained_variance,
        )


__all__ = ["PCA", "compute_pca", "principal_components"]
