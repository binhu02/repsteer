from __future__ import annotations

import weakref
from typing import Any

from repsteer.core.errors import PlanCompilationError
from repsteer.core.intervention import as_plan

from .compiled_plan import CompiledIntervention, CompiledPlan


def _validate_protocol(value: Any, method: str, role: str, index: int) -> None:
    if not callable(getattr(value, method, None)):
        raise PlanCompilationError(
            f"intervention {index} {role} ({type(value).__name__}) must implement "
            f"{method}()"
        )


def _assert_compatibility(
    artifact: Any,
    model: Any,
    site: Any,
    hidden_size: int,
    compatibility: str,
) -> str:
    try:
        from repsteer.artifacts.compatibility import assert_compatible

        result = assert_compatible(
            artifact,
            model,
            compatibility=compatibility,
            site=site,
            hidden_size=hidden_size,
        )
        return result.level.value
    except ImportError:
        # Defensive fallback for minimal installations that use protocol-only
        # artifacts.  Hidden dimension and exact identity remain safe-fail.
        metadata = getattr(artifact, "metadata", None)
        artifact_hidden = getattr(metadata, "hidden_size", None)
        if artifact_hidden != hidden_size:
            raise PlanCompilationError(
                f"artifact hidden size {artifact_hidden} does not match resolved "
                f"site hidden size {hidden_size}"
            )
        if compatibility == "exact":
            artifact_id = getattr(metadata, "model_id", None)
            artifact_revision = getattr(metadata, "model_revision", None)
            if artifact_id != model.model_id or artifact_revision != model.revision:
                raise PlanCompilationError(
                    "exact artifact identity mismatch: "
                    f"{artifact_id}@{artifact_revision} != "
                    f"{model.model_id}@{model.revision}"
                )
        return compatibility


def compile_plan(
    model: Any,
    plan: Any,
    *,
    compatibility: str = "exact",
) -> CompiledPlan:
    """Resolve and validate a plan without registering any PyTorch hooks."""

    if isinstance(plan, CompiledPlan):
        source = plan.source_model
        if source is not None and source is not model:
            raise PlanCompilationError(
                "compiled plan belongs to a different HFSteerableModel instance"
            )
        if plan.model_id != model.model_id or plan.revision != model.revision:
            raise PlanCompilationError(
                "compiled plan model identity no longer matches the target wrapper"
            )
        return plan

    try:
        normalized = as_plan(plan)
    except (TypeError, ValueError) as exc:
        raise PlanCompilationError(f"invalid SteeringPlan: {exc}") from exc

    ordered_pairs = sorted(
        enumerate(normalized.interventions),
        key=lambda pair: (pair[1].priority, pair[0]),
    )
    compiled: list[CompiledIntervention] = []
    warnings: list[str] = []
    for execution_index, (declaration_index, intervention) in enumerate(ordered_pairs):
        try:
            site = intervention.resolved_site
        except (AttributeError, TypeError, ValueError) as exc:
            raise PlanCompilationError(
                f"intervention {declaration_index} has no resolvable site: {exc}"
            ) from exc
        try:
            resolved = model.resolve_site(site)
        except Exception as exc:
            if isinstance(exc, PlanCompilationError):
                raise
            raise PlanCompilationError(
                f"failed to resolve intervention {declaration_index} at {site}: {exc}"
            ) from exc

        _validate_protocol(
            intervention.positions, "select", "positions", declaration_index
        )
        _validate_protocol(
            intervention.strength, "value", "strength", declaration_index
        )
        _validate_protocol(intervention.gate, "evaluate", "gate", declaration_index)
        _validate_protocol(
            intervention.operator, "apply", "operator", declaration_index
        )
        if intervention.phase not in ("prefill", "decode", "both"):
            raise PlanCompilationError(
                f"intervention {declaration_index} has invalid phase "
                f"{intervention.phase!r}"
            )

        metadata = getattr(intervention.artifact, "metadata", None)
        source_site = getattr(metadata, "site", None)
        explicit_cross_site = intervention.site is not None and source_site != site
        # Explicitly targeting another site is a supported, visible research
        # decision.  Exact still binds model id + revision + hidden dimension;
        # it does not pretend the target is the artifact's capture site.
        compatibility_site = source_site if explicit_cross_site else site
        actual_compatibility = _assert_compatibility(
            intervention.artifact,
            model,
            compatibility_site,
            resolved.hidden_dim,
            compatibility,
        )
        item_warnings: list[str] = []
        if explicit_cross_site:
            item_warnings.append(
                f"explicit cross-site application: learned at {source_site}, applied at {site}"
            )
        artifact_dtype = getattr(metadata, "dtype", None)
        model_dtype = getattr(model, "dtype", None)
        if (
            artifact_dtype
            and model_dtype
            and str(model_dtype).removeprefix("torch.") != artifact_dtype
        ):
            item_warnings.append(
                f"artifact dtype {artifact_dtype} will be converted at runtime to {model_dtype}"
            )
        compiled.append(
            CompiledIntervention(
                declaration_index=declaration_index,
                execution_index=execution_index,
                intervention=intervention,
                site=site,
                resolved_site=resolved,
                compatibility=actual_compatibility,
                warnings=tuple(item_warnings),
            )
        )

    return CompiledPlan(
        interventions=tuple(compiled),
        model_id=model.model_id,
        revision=model.revision,
        architecture=model.architecture,
        compatibility=compatibility,
        warnings=tuple(warnings),
        _model_ref=weakref.ref(model),
    )


__all__ = ["compile_plan"]
