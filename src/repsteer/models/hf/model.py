from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor, nn

from repsteer.core.errors import (
    GenerationPhaseError,
    HookLifecycleError,
    MissingOptionalDependencyError,
    PositionResolutionError,
    ProcessorCompatibilityError,
)
from repsteer.positions.base import validate_mask
from repsteer.runtime.compiled_plan import CompiledPlan
from repsteer.runtime.compiler import compile_plan
from repsteer.runtime.hook_manager import HookManager
from repsteer.runtime.session import SteeringSession

from ..outputs import GenerationResult
from .adapters import ArchitectureAdapter, get_adapter
from .generation import GenerationTracker


def _torch_dtype(value: Any) -> Any:
    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower().removeprefix("torch.").replace("-", "")
    aliases = {
        "auto": "auto",
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype {value!r}")
    return aliases[normalized]


def _transformers() -> Any:
    try:
        import transformers

        return transformers
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "Hugging Face model loading requires the optional 'transformers' "
            "dependency; install repsteer[hf]"
        ) from exc


def _parameter_device(model: nn.Module) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        try:
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if isinstance(weight, Tensor) and weight.device.type != "meta":
                return weight.device
        except (AttributeError, RuntimeError):
            pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _parameter_dtype(model: nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.get_default_dtype()


def _move_mapping(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device) if isinstance(value, Tensor) else value
        for key, value in values.items()
    }


def _batch_size_from_text(value: Any) -> int:
    if isinstance(value, str):
        return 1
    if isinstance(value, Sequence):
        return len(value)
    return 1


def _image_cardinality(
    value: Any,
) -> tuple[int, tuple[int, ...] | None]:
    """Return total images and optional per-prompt counts.

    A flat sequence follows the Transformers processor-order convention. A
    nested list/tuple is treated as an explicit prompt batch, so its per-prompt
    cardinality can be validated before processor-specific placeholder
    expansion. Array/tensor batches use their leading dimension.
    """

    ndim = getattr(value, "ndim", None)
    shape = getattr(value, "shape", None)
    if isinstance(ndim, int):
        if ndim <= 3:
            return 1, None
        if ndim == 4 and shape is not None:
            return int(shape[0]), None
        raise PositionResolutionError(
            "static image input must be one image or a rank-4 image batch"
        )
    if isinstance(value, (list, tuple)):
        if not value:
            return 0, None
        if all(isinstance(item, (list, tuple)) for item in value):
            per_prompt = tuple(len(item) for item in value)
            return sum(per_prompt), per_prompt
        return len(value), None
    return 1, None


def _component_identity(value: Any) -> tuple[str | None, str | None]:
    """Best-effort id/revision extraction for tokenizer/processor validation."""

    if value is None:
        return None, None
    candidates = (
        value,
        getattr(value, "config", None),
        getattr(value, "image_processor", None),
        getattr(value, "tokenizer", None),
    )
    identifier: Any | None = None
    revision: Any | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        identifier = identifier or getattr(candidate, "name_or_path", None)
        identifier = identifier or getattr(candidate, "_name_or_path", None)
        revision = revision or getattr(candidate, "_commit_hash", None)
        revision = revision or getattr(candidate, "revision", None)
        init_kwargs = getattr(candidate, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            revision = revision or init_kwargs.get("_commit_hash")
            revision = revision or init_kwargs.get("revision")
            identifier = identifier or init_kwargs.get("name_or_path")
    return (
        None if identifier is None else str(identifier),
        None if revision is None else str(revision),
    )


def _extract_sequences(raw: Any) -> Tensor:
    if isinstance(raw, Tensor):
        return raw
    sequences = getattr(raw, "sequences", None)
    if isinstance(sequences, Tensor):
        return sequences
    if isinstance(raw, Mapping) and isinstance(raw.get("sequences"), Tensor):
        return cast(Tensor, raw["sequences"])
    raise TypeError(
        f"Transformers generate returned {type(raw).__name__} without tensor sequences"
    )


@contextmanager
def _seeded(seed: int | None, model: nn.Module) -> Iterator[None]:
    if seed is None:
        yield
        return
    cuda_devices: list[int] = []
    for parameter in model.parameters():
        if parameter.device.type == "cuda" and parameter.device.index is not None:
            if parameter.device.index not in cuda_devices:
                cuda_devices.append(parameter.device.index)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        yield


@contextmanager
def _capture_grad(enabled: bool) -> Iterator[None]:
    if enabled:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            yield
    else:
        with torch.inference_mode():
            yield


class HFSteerableModel:
    """Activation-steerable wrapper around an eager Hugging Face causal LM."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        *,
        processor: Any | None = None,
        adapter: ArchitectureAdapter | None = None,
        model_id: str | None = None,
        revision: str | None = None,
        processor_id: str | None = None,
        processor_revision: str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self.processor = processor
        self.adapter = adapter or get_adapter(model)
        config = getattr(model, "config", None)
        inferred_id = getattr(config, "_name_or_path", None) or getattr(
            model, "name_or_path", None
        )
        self.model_id = str(model_id or inferred_id or type(model).__qualname__)
        inferred_revision = getattr(config, "_commit_hash", None) or getattr(
            config, "revision", None
        )
        self.revision = (
            str(revision or inferred_revision)
            if (revision or inferred_revision)
            else None
        )
        inferred_processor_id, inferred_processor_revision = _component_identity(
            processor
        )
        self.processor_id = (
            str(processor_id or inferred_processor_id)
            if (processor_id or inferred_processor_id)
            else None
        )
        self.processor_revision = (
            str(processor_revision or inferred_processor_revision)
            if (processor_revision or inferred_processor_revision)
            else None
        )
        if (
            processor is not None
            and self.revision is not None
            and self.processor_revision is not None
            and self.revision != self.processor_revision
        ):
            raise ProcessorCompatibilityError(
                "processor/model revision mismatch: "
                f"processor {self.processor_id or '<unknown>'}"
                f"@{self.processor_revision} != model {self.model_id}@{self.revision}; "
                "load the processor from the same pinned revision"
            )
        architectures = getattr(config, "architectures", None)
        self.architecture = (
            str(architectures[0])
            if isinstance(architectures, (tuple, list)) and architectures
            else type(model).__qualname__
        )
        self.generation_tracker = GenerationTracker()
        self.hook_manager = HookManager(self)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> "HFSteerableModel":
        return cls(model, tokenizer, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> "HFSteerableModel":
        return from_pretrained(model_name_or_path, **kwargs)

    @property
    def raw_model(self) -> nn.Module:
        return self.model

    @property
    def config(self) -> Any:
        return getattr(self.model, "config", None)

    @property
    def name_or_path(self) -> str:
        return self.model_id

    @property
    def hidden_size(self) -> int:
        return self.adapter.hidden_size(self.model)

    @property
    def device(self) -> torch.device:
        return _parameter_device(self.model)

    @property
    def dtype(self) -> torch.dtype:
        return _parameter_dtype(self.model)

    def resolve_site(self, site: Any) -> Any:
        return self.adapter.resolve(self.model, site)

    def compile(self, plan: Any, *, compatibility: str = "exact") -> CompiledPlan:
        return compile_plan(self, plan, compatibility=compatibility)

    def steer(
        self,
        plan: Any,
        *,
        compatibility: str = "exact",
    ) -> SteeringSession:
        compiled = self.compile(plan, compatibility=compatibility)
        return SteeringSession(self.hook_manager, compiled)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.hook_manager.assert_owner()
        return self.model(*args, **kwargs)

    __call__ = forward

    def _encode_texts(
        self,
        texts: Any,
        *,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.tokenizer is None:
            raise MissingOptionalDependencyError(
                "string inputs require a tokenizer; pass tokenizer=... to from_model"
            )
        options = dict(tokenizer_kwargs or {})
        options.setdefault("return_tensors", "pt")
        options.setdefault("padding", _batch_size_from_text(texts) > 1)
        if (
            options.get("padding")
            and getattr(self.tokenizer, "pad_token_id", None) is None
        ):
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None and hasattr(self.tokenizer, "pad_token"):
                self.tokenizer.pad_token = eos_token
        encoded = self.tokenizer(texts, **options)
        if not isinstance(encoded, Mapping):
            try:
                encoded = dict(encoded)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tokenizer must return a mapping of model inputs"
                ) from exc
        return _move_mapping(encoded, self.device)

    def _encode_multimodal(
        self,
        texts: Any,
        images: Any,
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.processor is None:
            raise MissingOptionalDependencyError(
                "image inputs require a multimodal processor; pass processor=... "
                "to from_model or load a VLM with from_pretrained"
            )
        texts = self._prepare_multimodal_texts(texts, images)
        options = dict(processor_kwargs or {})
        options.setdefault("return_tensors", "pt")
        options.setdefault("padding", _batch_size_from_text(texts) > 1)
        encoded = self.processor(text=texts, images=images, **options)
        if not isinstance(encoded, Mapping):
            try:
                encoded = dict(encoded)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "multimodal processor must return a mapping of model inputs"
                ) from exc
        return _move_mapping(encoded, self.device)

    def _prepare_multimodal_texts(self, texts: Any, images: Any) -> Any:
        """Inject one unambiguous image placeholder for built-in VLMs.

        Real Qwen2.5-VL and InternVL processors expand their ``image_token`` in
        the prompt. Automatic insertion is safe only for one prompt and one
        image. Multi-image and prompt-batch callers must place the processor's
        token explicitly so image ordering is visible and verifiable.
        """

        architecture = getattr(self.adapter, "architecture_name", "")
        if architecture not in {"qwen2_5_vl", "internvl"}:
            return texts
        processor = self.processor
        image_token = getattr(processor, "image_token", None)
        if not isinstance(image_token, str) or not image_token:
            raise ProcessorCompatibilityError(
                f"{architecture} processor must expose a non-empty image_token"
            )
        single_prompt = isinstance(texts, str)
        tuple_batch = isinstance(texts, tuple)
        if single_prompt:
            prompts = [texts]
        elif isinstance(texts, Sequence) and all(
            isinstance(item, str) for item in texts
        ):
            prompts = list(texts)
        else:
            raise TypeError(
                "multimodal processor text must be a string or string batch"
            )

        image_count, per_prompt = _image_cardinality(images)
        if image_count <= 0:
            raise PositionResolutionError(
                "multimodal generation requires at least one static image"
            )
        if per_prompt is not None and len(per_prompt) != len(prompts):
            raise PositionResolutionError(
                "a nested images batch must contain one image list per prompt"
            )

        placeholder_counts = tuple(prompt.count(image_token) for prompt in prompts)
        explicit_count = sum(placeholder_counts)
        if explicit_count:
            if explicit_count != image_count:
                raise PositionResolutionError(
                    "image placeholder count does not match image input count: "
                    f"found {explicit_count} {image_token!r} placeholder(s) for "
                    f"{image_count} image(s)"
                )
            if per_prompt is not None and placeholder_counts != per_prompt:
                raise PositionResolutionError(
                    "per-prompt image placeholder counts do not match the nested "
                    "images batch"
                )
            return texts

        if len(prompts) != 1 or image_count != 1:
            raise PositionResolutionError(
                "automatic image placeholder insertion is only unambiguous for "
                "one prompt and one image; insert processor.image_token explicitly "
                "for multi-image or prompt-batch generation"
            )

        placeholder = image_token
        if architecture == "qwen2_5_vl":
            tokenizer = getattr(processor, "tokenizer", None)
            config = getattr(self.model, "config", None)

            def special_token(name: str, fallback: str) -> str:
                token_id = getattr(config, f"{name}_token_id", None)
                convert = getattr(tokenizer, "convert_ids_to_tokens", None)
                if token_id is not None and callable(convert):
                    value = convert(int(token_id))
                    content = getattr(value, "content", value)
                    if isinstance(content, str) and content:
                        return content
                return fallback

            start = special_token("vision_start", "<|vision_start|>")
            end = special_token("vision_end", "<|vision_end|>")
            placeholder = f"{start}{image_token}{end}"

        prompts[0] = f"{placeholder}\n{prompts[0]}"
        if single_prompt:
            return prompts[0]
        return tuple(prompts) if tuple_batch else prompts

    def _prepare_generation(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[
        tuple[Any, ...],
        dict[str, Any],
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Any | None,
    ]:
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", None)
        processor_kwargs = kwargs.pop("processor_kwargs", None)
        prompt = kwargs.pop("prompt", None)
        singular_image = kwargs.pop("image", None)
        plural_images = kwargs.pop("images", None)
        if singular_image is not None and plural_images is not None:
            raise TypeError("pass either image=... or images=..., not both")
        images = plural_images if plural_images is not None else singular_image
        if kwargs.get("video") is not None or kwargs.get("videos") is not None:
            raise NotImplementedError(
                "repsteer 0.2.0 supports static VLM images, not video modality maps"
            )
        values = args
        if prompt is not None:
            if values:
                raise TypeError("prompt was passed both positionally and by keyword")
            values = (prompt,)

        if images is not None:
            if not values or not (
                isinstance(values[0], str)
                or (
                    isinstance(values[0], Sequence)
                    and not isinstance(values[0], (Tensor, bytes, bytearray, Mapping))
                    and all(isinstance(item, str) for item in values[0])
                )
            ):
                raise TypeError(
                    "multimodal generation requires a string prompt or prompt batch"
                )
            if len(values) != 1:
                raise TypeError(
                    "multimodal generation accepts one prompt or prompt batch argument"
                )
            encoded = self._encode_multimodal(
                values[0], images, processor_kwargs=processor_kwargs
            )
            duplicates = set(encoded) & set(kwargs)
            if duplicates:
                raise TypeError(
                    f"processed inputs conflict with generation kwargs: "
                    f"{sorted(duplicates)}"
                )
            kwargs = {**encoded, **kwargs}
            values = ()
        elif values and (
            isinstance(values[0], str)
            or (
                isinstance(values[0], Sequence)
                and not isinstance(values[0], (Tensor, bytes, bytearray, Mapping))
                and all(isinstance(item, str) for item in values[0])
            )
        ):
            if len(values) != 1:
                raise TypeError(
                    "text generation accepts one prompt or prompt batch argument"
                )
            encoded = self._encode_texts(values[0], tokenizer_kwargs=tokenizer_kwargs)
            duplicates = set(encoded) & set(kwargs)
            if duplicates:
                raise TypeError(
                    f"encoded inputs conflict with generation kwargs: {sorted(duplicates)}"
                )
            kwargs = {**encoded, **kwargs}
            values = ()
        elif values and isinstance(values[0], Mapping):
            if len(values) != 1:
                raise TypeError(
                    "encoded input mappings must be the sole positional argument"
                )
            encoded = _move_mapping(values[0], self.device)
            duplicates = set(encoded) & set(kwargs)
            if duplicates:
                raise TypeError(
                    f"encoded inputs conflict with generation kwargs: {sorted(duplicates)}"
                )
            kwargs = {**encoded, **kwargs}
            values = ()
        elif values and isinstance(values[0], Tensor):
            values = (values[0].to(device=self.device), *values[1:])
            kwargs = _move_mapping(kwargs, self.device)
        else:
            kwargs = _move_mapping(kwargs, self.device)

        input_ids = kwargs.get("input_ids")
        if (
            not isinstance(input_ids, Tensor)
            and values
            and isinstance(values[0], Tensor)
        ):
            input_ids = values[0]
        if not isinstance(input_ids, Tensor) and isinstance(
            kwargs.get("inputs"), Tensor
        ):
            input_ids = kwargs["inputs"]
        inputs_embeds = kwargs.get("inputs_embeds")
        attention_mask = kwargs.get("attention_mask")
        modality_inputs = kwargs
        if isinstance(input_ids, Tensor) and not isinstance(
            kwargs.get("input_ids"), Tensor
        ):
            modality_inputs = {**kwargs, "input_ids": input_ids}
        modality_map = self.adapter.build_modality_map(
            modality_inputs, model=self.model
        )
        return (
            values,
            kwargs,
            input_ids if isinstance(input_ids, Tensor) else None,
            inputs_embeds if isinstance(inputs_embeds, Tensor) else None,
            attention_mask if isinstance(attention_mask, Tensor) else None,
            modality_map,
        )

    def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
        self.hook_manager.assert_owner()
        seed = kwargs.pop("seed", None)
        decode_kwargs = kwargs.pop("decode_kwargs", None) or {}
        (
            values,
            generation_kwargs,
            input_ids,
            inputs_embeds,
            attention_mask,
            modality_map,
        ) = self._prepare_generation(args, kwargs)
        was_training = self.model.training
        self.model.eval()
        try:
            with self.generation_tracker.generation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                modality_map=modality_map,
            ):
                with (
                    _seeded(None if seed is None else int(seed), self.model),
                    torch.inference_mode(),
                ):
                    raw = self.model.generate(*values, **generation_kwargs)
        finally:
            self.model.train(was_training)
        sequences = _extract_sequences(raw)
        text: str | list[str] | None = None
        if self.tokenizer is not None:
            decoded = self.tokenizer.batch_decode(sequences, **decode_kwargs)
            text = decoded[0] if len(decoded) == 1 else list(decoded)
        return GenerationResult(text=text, token_ids=sequences, raw=raw)

    def _render_capture_inputs(self, request: Any) -> Any:
        inputs = list(request.inputs)
        if (
            len(inputs) == 1
            and isinstance(inputs[0], Mapping)
            and any(isinstance(value, Tensor) for value in inputs[0].values())
        ):
            return inputs
        apply_template = request.apply_chat_template
        first = inputs[0] if inputs else None
        structured = bool(
            isinstance(first, Mapping) and {"role", "content"} <= set(first)
        ) or bool(
            isinstance(first, Sequence)
            and not isinstance(first, (str, bytes, bytearray))
            and first
            and isinstance(first[0], Mapping)
            and {"role", "content"} <= set(first[0])
        )
        if not (apply_template is True or (apply_template is None and structured)):
            return inputs
        if self.tokenizer is None or not callable(
            getattr(self.tokenizer, "apply_chat_template", None)
        ):
            raise MissingOptionalDependencyError(
                "chat capture inputs require tokenizer.apply_chat_template"
            )
        rendered: list[str] = []
        for conversation in inputs:
            messages = conversation
            if request.system_prompt:
                if isinstance(messages, Mapping):
                    messages = [messages]
                messages = [
                    {"role": "system", "content": request.system_prompt},
                    *list(messages),
                ]
            rendered.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=request.add_generation_prompt,
                )
            )
        return rendered

    def _prepare_capture_inputs(
        self, request: Any, *, rendered_inputs: Any | None = None
    ) -> dict[str, Any]:
        values = (
            self._render_capture_inputs(request)
            if rendered_inputs is None
            else rendered_inputs
        )
        if (
            len(values) == 1
            and isinstance(values[0], Mapping)
            and any(isinstance(value, Tensor) for value in values[0].values())
        ):
            return _move_mapping(values[0], self.device)
        tokenizer_kwargs: dict[str, Any] = {}
        if isinstance(request.special_tokens, bool):
            tokenizer_kwargs["add_special_tokens"] = request.special_tokens
        needs_offsets = type(request.positions).__name__ == "TextSpan"
        if needs_offsets:
            tokenizer_kwargs["return_offsets_mapping"] = True
        try:
            return self._encode_texts(values, tokenizer_kwargs=tokenizer_kwargs)
        except (NotImplementedError, TypeError) as exc:
            if not needs_offsets:
                raise
            raise PositionResolutionError(
                "TextSpan capture requires a tokenizer that supports "
                "return_offsets_mapping (normally a Hugging Face fast tokenizer)"
            ) from exc

    def capture(self, request: Any) -> Any:
        from repsteer.capture.request import ActivationBatch

        self.hook_manager.assert_owner()
        if self.hook_manager.active:
            raise HookLifecycleError(
                "capture() cannot run inside an active steering session because "
                "the resulting activations and cache key would be ambiguous"
            )
        if self.generation_tracker.generation_active:
            raise GenerationPhaseError(
                "capture() cannot run inside an active generate() call"
            )
        resolved = self.resolve_site(request.site)
        rendered_inputs = self._render_capture_inputs(request)
        batch = self._prepare_capture_inputs(request, rendered_inputs=rendered_inputs)
        token_offsets = batch.pop("offset_mapping", None)
        input_ids = batch.get("input_ids")
        captured: list[Tensor] = []

        def capture_pre(module: Any, args: tuple[Any, ...]) -> None:
            del module
            captured.append(resolved.read(args))
            return None

        def capture_forward(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            captured.append(resolved.read(output))
            return None

        hook = (
            resolved.module.register_forward_pre_hook(capture_pre)
            if resolved.hook_kind == "forward_pre"
            else resolved.module.register_forward_hook(capture_forward)
        )
        was_training = self.model.training
        if request.model_mode == "eval":
            self.model.eval()
        elif request.model_mode == "train":
            self.model.train()
        else:
            hook.remove()
            raise ValueError("CaptureRequest.model_mode must be 'eval' or 'train'")

        manually_tracked = not self.hook_manager.active
        if manually_tracked:
            modality_map = self.adapter.build_modality_map(batch, model=self.model)
            self.generation_tracker.update((), batch, modality_map=modality_map)
        forward_completed = False
        try:
            with _seeded(request.seed, self.model), _capture_grad(request.gradient):
                self.model(**batch)
            forward_completed = True
        finally:
            hook.remove()
            self.model.train(was_training)
            if manually_tracked and not forward_completed:
                self.generation_tracker.current = None
        if not captured:
            raise RuntimeError(f"capture hook at {resolved.module_path} did not run")
        if len(captured) != 1:
            if manually_tracked:
                self.generation_tracker.current = None
            raise RuntimeError(
                f"capture hook at {resolved.module_path} ran {len(captured)} times; "
                "re-entrant/checkpointed forwards require an explicit capture policy"
            )

        activation = captured[0]
        context = self.generation_tracker.for_activation(
            activation,
            stream=getattr(request.site, "stream", "language"),
            component=getattr(request.site, "component", None),
        )
        texts = (
            tuple(rendered_inputs)
            if isinstance(rendered_inputs, Sequence)
            and not isinstance(rendered_inputs, (str, bytes, bytearray, Mapping))
            and all(isinstance(value, str) for value in rendered_inputs)
            else None
        )
        context = context.with_updates(texts=texts, token_offsets=token_offsets)
        if manually_tracked:
            self.generation_tracker.current = None
        raw_mask = request.positions.select(activation, context)
        mask = validate_mask(torch.as_tensor(raw_mask), activation, context)
        token_ids = input_ids if isinstance(input_ids, Tensor) else None
        selected, selected_mask, selected_ids = _select_capture(
            activation, mask, token_ids
        )
        if request.detach:
            selected = selected.detach()
        requested_dtype = _torch_dtype(request.dtype)
        if requested_dtype not in (None, "auto"):
            selected = selected.to(dtype=requested_dtype)
        sample_weights = (
            torch.as_tensor(
                request.sample_weights, dtype=torch.float32, device=selected.device
            )
            if request.sample_weights is not None
            else None
        )
        return ActivationBatch(
            activations=selected,
            request=request,
            attention_mask=selected_mask,
            token_ids=selected_ids,
            sample_weights=sample_weights,
            metadata={
                "resolved_module_path": resolved.module_path,
                "resolved_hook_kind": resolved.hook_kind,
                "architecture": resolved.architecture_name,
                "processor_id": self.processor_id,
                "processor_revision": self.processor_revision,
            },
            pooled=False,
        )

    capture_activations = capture

    def close(self) -> None:
        if self.hook_manager.active:
            self.hook_manager.clear()


def _select_capture(
    activation: Tensor,
    mask: Tensor,
    token_ids: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    unbatched = activation.ndim == 2
    working = activation.unsqueeze(0) if unbatched else activation
    if working.ndim < 3:
        raise PositionResolutionError(
            f"captured activation must be [B,S,H] or [S,H], got {tuple(activation.shape)}"
        )
    counts = mask.sum(dim=-1)
    equal_counts = counts.numel() <= 1 or bool(torch.all(counts == counts[0]).item())
    if equal_counts:
        count = int(counts[0].item()) if counts.numel() else 0
        selected = working[mask].reshape(working.shape[0], count, *working.shape[2:])
        selected_mask = torch.ones(
            (working.shape[0], count), dtype=torch.bool, device=working.device
        )
        selected_ids = None
        if (
            token_ids is not None
            and token_ids.ndim == 2
            and token_ids.shape == mask.shape
        ):
            selected_ids = token_ids.to(mask.device)[mask].reshape(
                working.shape[0], count
            )
    else:
        # Ragged selections remain padded; the selector mask is propagated to
        # the capture pooler, so unselected padding cannot affect the learner.
        selected = working
        selected_mask = mask
        selected_ids = token_ids
    # ActivationBatch always retains the batch axis inserted for unbatched sites.
    return selected, selected_mask, selected_ids


def from_model(
    model: nn.Module,
    tokenizer: Any | None = None,
    **kwargs: Any,
) -> HFSteerableModel:
    return HFSteerableModel.from_model(model, tokenizer, **kwargs)


def from_pretrained(
    model_name_or_path: str | nn.Module | None = None,
    *,
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    processor: Any | None = None,
    processor_id: str | None = None,
    processor_revision: str | None = None,
    adapter: ArchitectureAdapter | None = None,
    revision: str | None = None,
    dtype: str | torch.dtype | None = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
    multimodal: bool | None = None,
    model_class: Any | None = None,
    **model_kwargs: Any,
) -> HFSteerableModel:
    """Load a text/VLM model lazily, or wrap an already-constructed model."""

    processor_supplied = processor is not None
    if model is not None:
        if model_name_or_path is not None:
            raise TypeError("pass either model=... or model_name_or_path, not both")
        return from_model(
            model,
            tokenizer,
            processor=processor,
            processor_id=processor_id,
            processor_revision=processor_revision,
            adapter=adapter,
            revision=revision,
        )
    if isinstance(model_name_or_path, nn.Module):
        return from_model(
            model_name_or_path,
            tokenizer,
            processor=processor,
            processor_id=processor_id,
            processor_revision=processor_revision,
            adapter=adapter,
            revision=revision,
        )
    if not isinstance(model_name_or_path, str) or not model_name_or_path:
        raise TypeError("from_pretrained requires a model id/path or model=nn.Module")

    transformers = _transformers()
    resolved_dtype = _torch_dtype(dtype)
    # repsteer exposes the forward-compatible public spelling ``dtype`` while
    # supporting Transformers 4.49 through 4.x.  The older loading keyword is
    # accepted throughout that declared range; early 4.x releases would pass
    # an unknown ``dtype`` through to the model constructor instead.
    legacy_dtype = model_kwargs.pop("torch_dtype", None)
    if legacy_dtype is not None:
        if dtype is not None:
            raise TypeError("pass either dtype=... or torch_dtype=..., not both")
        resolved_dtype = _torch_dtype(legacy_dtype)
    if resolved_dtype is not None:
        model_kwargs.setdefault("torch_dtype", resolved_dtype)
    if revision is not None:
        model_kwargs.setdefault("revision", revision)

    config = model_kwargs.get("config")
    if config is None:
        config_options: dict[str, Any] = {}
        if revision is not None:
            config_options["revision"] = revision
        if "trust_remote_code" in model_kwargs:
            config_options["trust_remote_code"] = model_kwargs["trust_remote_code"]
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, **config_options
        )
        model_kwargs["config"] = config
    inferred_multimodal = bool(
        getattr(config, "vision_config", None) is not None
        or getattr(config, "visual", None) is not None
        or "vl" in str(getattr(config, "model_type", "")).lower()
    )
    use_multimodal = inferred_multimodal if multimodal is None else bool(multimodal)
    loader = model_class
    if loader is None:
        loader = (
            getattr(
                transformers,
                "AutoModelForImageTextToText",
                transformers.AutoModelForImageTextToText,
            )
            if use_multimodal
            else transformers.AutoModelForCausalLM
        )
    raw_model = loader.from_pretrained(model_name_or_path, **model_kwargs)
    commit = getattr(getattr(raw_model, "config", None), "_commit_hash", None)
    if use_multimodal and processor is None:
        options = dict(processor_kwargs or tokenizer_kwargs or {})
        if revision is not None:
            options.setdefault("revision", revision)
        processor = transformers.AutoProcessor.from_pretrained(
            model_name_or_path, **options
        )
    if tokenizer is None and processor is not None:
        tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        options = dict(tokenizer_kwargs or {})
        if revision is not None:
            options.setdefault("revision", revision)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, **options
        )
    return HFSteerableModel(
        raw_model,
        tokenizer,
        processor=processor,
        adapter=adapter,
        model_id=model_name_or_path,
        revision=commit or revision,
        processor_id=(
            processor_id
            or (
                model_name_or_path
                if processor is not None and not processor_supplied
                else None
            )
        ),
        processor_revision=(
            processor_revision
            or (
                commit or revision
                if processor is not None and not processor_supplied
                else None
            )
        ),
    )


__all__ = ["HFSteerableModel", "from_model", "from_pretrained"]
