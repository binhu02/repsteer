"""Safe safetensors + JSON artifact bundle I/O."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from repsteer.core.errors import (
    ArtifactFormatError,
    MissingOptionalDependencyError,
    UnsafeArtifactError,
)

from .base import (
    Artifact,
    ArtifactMetadata,
    DirectionArtifact,
    ProbeArtifact,
    SteeringArtifact,
    SubspaceArtifact,
)

MANIFEST_FILENAME = "manifest.json"
TENSORS_FILENAME = "tensors.safetensors"
CHECKSUMS_FILENAME = "checksums.json"


def _safetensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - dependency contract guard
        raise MissingOptionalDependencyError(
            "Artifact I/O requires safetensors; install the core repsteer dependencies"
        ) from exc
    return load_file, save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactFormatError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactFormatError(f"Expected a JSON object in {path}")
    return value


def save_artifact(artifact: SteeringArtifact, path: str | os.PathLike[str]) -> Path:
    """Write an artifact bundle and checksums without pickle or executable code."""

    if not isinstance(artifact, SteeringArtifact):
        raise TypeError("save_artifact expects a SteeringArtifact")
    destination = Path(path)
    if destination.exists() and not destination.is_dir():
        raise ArtifactFormatError(f"Artifact path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_FILENAME
    tensors_path = destination / TENSORS_FILENAME
    checksums_path = destination / CHECKSUMS_FILENAME

    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in artifact.tensors().items()
    }
    if not tensors:
        raise ArtifactFormatError("An artifact must contain at least one tensor")
    if not all(isinstance(name, str) and name for name in tensors):
        raise ArtifactFormatError("Artifact tensor names must be non-empty strings")
    _, save_file = _safetensors()
    temporary_tensors = tensors_path.with_name(f".{tensors_path.name}.tmp")
    save_file(tensors, str(temporary_tensors))
    os.replace(temporary_tensors, tensors_path)

    manifest = artifact.metadata.to_manifest()
    manifest["tensor_keys"] = sorted(tensors)
    manifest["artifact_fingerprint"] = artifact.fingerprint()
    _write_json(manifest_path, manifest)
    checksums = {
        "algorithm": "sha256",
        "files": {
            MANIFEST_FILENAME: _sha256(manifest_path),
            TENSORS_FILENAME: _sha256(tensors_path),
        },
    }
    _write_json(checksums_path, checksums)
    return destination


def verify_artifact_checksums(path: str | os.PathLike[str]) -> None:
    source = Path(path)
    checksums_path = source / CHECKSUMS_FILENAME
    if not checksums_path.is_file():
        raise UnsafeArtifactError(f"Artifact bundle has no {CHECKSUMS_FILENAME}")
    document = _read_json(checksums_path)
    if document.get("algorithm") != "sha256":
        raise UnsafeArtifactError(
            f"Unsupported checksum algorithm {document.get('algorithm')!r}"
        )
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise UnsafeArtifactError("checksums.json has no valid 'files' mapping")
    required = {MANIFEST_FILENAME, TENSORS_FILENAME}
    if not required.issubset(files):
        missing = ", ".join(sorted(required - set(files)))
        raise UnsafeArtifactError(f"Artifact checksums are missing: {missing}")
    for filename in required:
        expected = str(files[filename]).removeprefix("sha256:")
        file_path = source / filename
        if not file_path.is_file():
            raise UnsafeArtifactError(f"Artifact bundle is missing {filename}")
        actual = _sha256(file_path)
        if not hmac.compare_digest(actual, expected):
            raise UnsafeArtifactError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )


def _build_artifact(
    metadata: ArtifactMetadata, tensors: Mapping[str, torch.Tensor]
) -> Artifact:
    artifact_type = metadata.artifact_type
    if artifact_type == "direction":
        if "direction" not in tensors:
            raise ArtifactFormatError("Direction artifact has no 'direction' tensor")
        return DirectionArtifact(metadata, tensors["direction"])
    if artifact_type == "subspace":
        if "basis" not in tensors:
            raise ArtifactFormatError("Subspace artifact has no 'basis' tensor")
        return SubspaceArtifact(
            metadata,
            tensors["basis"],
            tensors.get("mean"),
            tensors.get("explained_variance"),
        )
    if artifact_type == "probe":
        if "weight" not in tensors:
            raise ArtifactFormatError("Probe artifact has no 'weight' tensor")
        return ProbeArtifact(metadata, tensors["weight"], tensors.get("bias"))
    raise ArtifactFormatError(f"Unsupported artifact_type {artifact_type!r}")


def load_artifact(
    path: str | os.PathLike[str],
    *,
    verify_checksums: bool = True,
    device: str | torch.device = "cpu",
) -> Artifact:
    """Load a bundle after integrity verification (enabled by default)."""

    source = Path(path)
    if not source.is_dir():
        raise ArtifactFormatError(f"Artifact path is not a directory: {source}")
    if verify_checksums:
        verify_artifact_checksums(source)
    manifest = _read_json(source / MANIFEST_FILENAME)
    schema_version = str(manifest.get("schema_version", ""))
    if schema_version.split(".", 1)[0] != "1":
        raise ArtifactFormatError(
            f"Unsupported artifact schema {schema_version!r}; this version reads 1.x"
        )
    metadata = ArtifactMetadata.from_dict(manifest)
    load_file, _ = _safetensors()
    try:
        tensors = load_file(str(source / TENSORS_FILENAME), device=str(device))
    except Exception as exc:
        raise ArtifactFormatError(f"Cannot load artifact tensors: {exc}") from exc
    expected_keys = manifest.get("tensor_keys")
    if expected_keys is not None and set(expected_keys) != set(tensors):
        raise ArtifactFormatError(
            "Manifest tensor_keys do not match tensors.safetensors: "
            f"{sorted(expected_keys)} != {sorted(tensors)}"
        )
    artifact = _build_artifact(metadata, tensors)
    expected_fingerprint = manifest.get("artifact_fingerprint")
    if expected_fingerprint and artifact.fingerprint() != expected_fingerprint:
        raise UnsafeArtifactError(
            "Artifact content fingerprint does not match the manifest"
        )
    return artifact


__all__ = [
    "CHECKSUMS_FILENAME",
    "MANIFEST_FILENAME",
    "TENSORS_FILENAME",
    "load_artifact",
    "save_artifact",
    "verify_artifact_checksums",
]
