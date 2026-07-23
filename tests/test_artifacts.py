from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from repsteer.artifacts import (
    ArtifactMetadata,
    DirectionArtifact,
    ProbeArtifact,
    SubspaceArtifact,
    load_artifact,
)
from repsteer.core import (
    ArtifactCompatibilityError,
    Site,
    SiteResolutionError,
    UnsafeArtifactError,
)


class _Target:
    model_id = "tiny/model"
    revision = "rev-a"
    architecture = "TinyForCausalLM"
    hidden_size = 3


def _artifact() -> DirectionArtifact:
    return DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny/model",
            model_revision="rev-a",
            architecture="TinyForCausalLM",
            site=Site("language", "resid_post", 0),
            hidden_size=3,
            method="diff_mean",
            dataset_fingerprint="sha256:data",
            seed=42,
        ),
        torch.tensor([1.0, -2.0, 3.0]),
    )


def test_artifact_save_load_roundtrip_is_lossless_and_non_pickle(tmp_path):
    artifact = _artifact()
    artifact.save(tmp_path)
    restored = load_artifact(tmp_path)

    assert isinstance(restored, DirectionArtifact)
    assert torch.equal(restored.direction, artifact.direction)
    assert restored.metadata.to_dict() == artifact.metadata.to_dict()
    assert restored.fingerprint() == artifact.fingerprint()
    assert {path.name for path in tmp_path.iterdir()} == {
        "checksums.json",
        "manifest.json",
        "tensors.safetensors",
    }


def test_artifact_checksum_and_revision_mismatch_fail_closed(tmp_path):
    artifact = _artifact()
    artifact.save(tmp_path)
    target = _Target()
    artifact.bind(target)

    target.revision = "rev-b"
    with pytest.raises(ArtifactCompatibilityError, match="model revision differs"):
        artifact.bind(target)

    tensors = tmp_path / "tensors.safetensors"
    payload = tensors.read_bytes()
    tensors.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(UnsafeArtifactError, match="Checksum mismatch"):
        load_artifact(tmp_path)


def test_artifact_container_is_frozen():
    artifact = _artifact()
    with pytest.raises(FrozenInstanceError):
        artifact.direction = torch.zeros(3)  # type: ignore[misc]


class _ResolvingTarget(_Target):
    def __init__(self, *, hidden_dim: int = 3) -> None:
        self.hidden_dim = hidden_dim
        self.resolved: list[Site] = []

    def resolve_site(self, site: Site):
        self.resolved.append(site)
        if site.component != "resid_post" or site.layer != 0:
            raise SiteResolutionError(f"cannot resolve {site}")
        return SimpleNamespace(site=site, hidden_dim=self.hidden_dim)


def test_bind_resolves_metadata_site_and_uses_its_hidden_dimension():
    artifact = _artifact()
    target = _ResolvingTarget()

    assert artifact.bind(target) is artifact
    assert target.resolved == [artifact.metadata.site]

    with pytest.raises(
        ArtifactCompatibilityError, match=r"hidden size differs \(3 != 4\)"
    ):
        artifact.bind(_ResolvingTarget(hidden_dim=4), compatibility="dimension")


@pytest.mark.parametrize(
    "site",
    [
        Site("language", "not_a_component", 0),
        Site("language", "resid_post", 99),
    ],
)
def test_bind_invalid_resolved_site_fails_before_identity_compatibility(site):
    artifact = _artifact().with_metadata(_artifact().metadata.with_updates(site=site))
    target = _ResolvingTarget()
    target.revision = "wrong-revision"

    with pytest.raises(SiteResolutionError, match="cannot resolve"):
        artifact.bind(target)


def test_artifact_metadata_dtype_must_match_primary_tensor():
    with pytest.raises(ValueError, match="metadata dtype 'float16'.*'float32'"):
        DirectionArtifact(
            ArtifactMetadata(dtype="torch.float16"),
            torch.ones(3, dtype=torch.float32),
        )

    artifact = DirectionArtifact(
        ArtifactMetadata(dtype="torch.float32"),
        torch.ones(3, dtype=torch.float32),
    )
    assert artifact.metadata.dtype == "float32"


def test_attached_tensors_follow_primary_dtype_and_device():
    basis = torch.ones((2, 3), dtype=torch.float64)
    subspace = SubspaceArtifact(
        ArtifactMetadata(),
        basis,
        mean=torch.ones(3, dtype=torch.float32),
        explained_variance=torch.ones(2, dtype=torch.float16),
    )
    assert subspace.mean is not None
    assert subspace.explained_variance is not None
    assert subspace.mean.dtype == basis.dtype
    assert subspace.explained_variance.dtype == basis.dtype
    assert subspace.mean.device == basis.device
    assert subspace.explained_variance.device == basis.device

    weight = torch.ones(3, dtype=torch.float16)
    probe = ProbeArtifact(
        ArtifactMetadata(),
        weight,
        bias=torch.tensor(1.0, dtype=torch.float64),
    )
    assert probe.bias is not None
    assert probe.bias.dtype == weight.dtype
    assert probe.bias.device == weight.device
