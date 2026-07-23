import pytest
import torch
import transformers

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact, load_artifact
from repsteer.capture import CaptureRequest
from repsteer.core import ArtifactCompatibilityError, HookLifecycleError, Intervention
from repsteer.data import ContrastivePairs
from repsteer.learners import DiffMean
from repsteer.models import from_model, from_pretrained
from repsteer.operators import Add
from repsteer.positions import AllTokens, GeneratedTokens, LastNonPaddingToken, TextSpan
from repsteer.schedules import Constant
from repsteer.sites import resid_post


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self):
        self.vocabulary = {
            "kind words": [1, 8, 9],
            "help people": [1, 10, 11],
            "cruel words": [1, 12, 9],
            "hurt people": [1, 13, 11],
        }

    def __call__(self, texts, *, return_tensors="pt", padding=False, **kwargs):
        assert return_tensors == "pt"
        values = [texts] if isinstance(texts, str) else list(texts)
        rows = [self.vocabulary.get(value, [1, 3, 4]) for value in values]
        width = max(map(len, rows))
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        result = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }
        if kwargs.get("return_offsets_mapping"):
            offsets = []
            for text, row in zip(values, rows, strict=True):
                cursor = 0
                current = [[0, 0]]
                for word in text.split():
                    start = text.index(word, cursor)
                    stop = start + len(word)
                    current.append([start, stop])
                    cursor = stop
                current.extend([[0, 0]] * (width - len(row)))
                offsets.append(current)
            result["offset_mapping"] = torch.tensor(offsets)
        return result

    def batch_decode(self, sequences, **_kwargs):
        return [" ".join(map(str, row.tolist())) for row in sequences]


def _wrapper():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    raw = transformers.LlamaForCausalLM(config).eval()
    return from_model(raw, _Tokenizer(), model_id="tiny/llama", revision="rev-1")


def test_public_dtype_uses_the_transformers_4x_loading_keyword():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    raw = transformers.LlamaForCausalLM(config).eval()

    class Loader:
        seen = None

        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            cls.seen = kwargs
            return raw

    wrapped = from_pretrained(
        "tiny/llama",
        config=config,
        model_class=Loader,
        tokenizer=_Tokenizer(),
        revision="rev-1",
        dtype="float32",
    )

    assert wrapped.raw_model is raw
    assert Loader.seen["torch_dtype"] is torch.float32
    assert "dtype" not in Loader.seen


def _artifact(wrapper, *, revision="rev-1"):
    site = resid_post(0)
    return DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=revision,
            architecture=wrapper.architecture,
            site=site,
            hidden_size=wrapper.hidden_size,
            method="unit",
        ),
        torch.linspace(-1, 1, wrapper.hidden_size),
    )


def test_compile_is_pure_zero_strength_is_baseline_and_exception_cleans_hooks():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    resolved = wrapper.resolve_site(artifact.metadata.site)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(0),
        phase="both",
    )
    ids = torch.tensor([[1, 5, 6]])
    baseline = wrapper(input_ids=ids).logits
    module_hooks = len(resolved.module._forward_hooks)

    compiled = wrapper.compile(control)
    assert len(resolved.module._forward_hooks) == module_hooks
    assert "model.layers.0" in compiled.explain()
    try:
        with wrapper.steer(compiled):
            # A statically zero intervention is removed from runtime hook groups,
            # including the otherwise-unneeded root phase tracker.
            assert wrapper.hook_manager.hook_count == 0
            steered = wrapper(input_ids=ids).logits
            assert torch.equal(steered, baseline)
            raise RuntimeError("body failure")
    except RuntimeError as error:
        assert str(error) == "body failure"

    assert wrapper.hook_manager.hook_count == 0
    assert len(resolved.module._forward_hooks) == module_hooks


def test_cached_generate_tracks_decode_and_zero_control_matches_baseline():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    ids = torch.tensor([[1, 5, 6]])
    attention_mask = torch.ones_like(ids)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0),
        phase="decode",
    )

    baseline = wrapper.generate(
        ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
        seed=42,
    )
    with wrapper.steer(control):
        steered = wrapper.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            seed=42,
        )

    assert torch.equal(steered.token_ids, baseline.token_ids)
    assert wrapper.hook_manager.hook_count == 0


def test_generation_tracker_exposes_decode_steps_to_operator():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)

    class RecordingAdd:
        def __init__(self):
            self.seen = []

        def apply(self, activation, artifact, strength, context):
            self.seen.append(
                (context.phase, context.generation_step, activation.shape[-2])
            )
            return Add().apply(activation, artifact, strength, context)

    operator = RecordingAdd()
    control = Intervention(
        artifact=artifact,
        operator=operator,
        positions=GeneratedTokens(),
        strength=Constant(0.25),
        phase="decode",
    )
    with wrapper.steer(control):
        wrapper.generate(
            torch.tensor([[1, 5, 6]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    assert operator.seen == [("decode", 0, 1), ("decode", 1, 1)]


def test_zero_strength_does_not_invoke_rng_consuming_custom_operator():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)

    class RandomConsumingOperator:
        def __init__(self):
            self.calls = 0

        def apply(self, activation, artifact, strength, context):
            del artifact, strength, context
            self.calls += 1
            torch.rand(1024, device=activation.device)
            return activation

    operator = RandomConsumingOperator()
    control = Intervention(
        artifact=artifact,
        operator=operator,
        positions=GeneratedTokens(),
        strength=Constant(0),
        phase="decode",
    )
    kwargs = {
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "min_new_tokens": 4,
        "max_new_tokens": 4,
        "do_sample": True,
        "seed": 123,
    }
    input_ids = torch.tensor([[1, 5, 6]])

    baseline = wrapper.generate(input_ids, **kwargs)
    with wrapper.steer(control):
        steered = wrapper.generate(input_ids, **kwargs)

    assert operator.calls == 0
    assert torch.equal(steered.token_ids, baseline.token_ids)


def test_diffmean_vertical_slice_capture_save_load_and_cross_layer_compile(tmp_path):
    wrapper = _wrapper()
    data = ContrastivePairs.from_records(
        [
            {"positive": "kind words", "negative": "cruel words"},
            {"positive": "help people", "negative": "hurt people"},
        ]
    )
    artifact = DiffMean(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
    ).fit(wrapper, data)

    assert artifact.direction.shape == (wrapper.hidden_size,)
    assert artifact.metadata.architecture == wrapper.architecture
    assert artifact.metadata.tokenizer["id"] == wrapper.model_id
    assert artifact.metadata.tokenizer["revision"] == wrapper.revision
    assert "torch" in artifact.metadata.provenance["environment"]["dependencies"]
    artifact.save(tmp_path)
    assert torch.equal(load_artifact(tmp_path).direction, artifact.direction)

    cross_layer = Intervention(
        artifact=artifact,
        site=resid_post(1),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(1),
        phase="decode",
    )
    explanation = wrapper.compile(cross_layer).explain(format="dict")
    assert "cross-site" in explanation["interventions"][0]["warnings"][0]


def test_revision_mismatch_fails_by_default():
    wrapper = _wrapper()
    control = Intervention(
        artifact=_artifact(wrapper, revision="other-revision"),
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(1),
    )

    with torch.no_grad():
        try:
            wrapper.compile(control)
        except ArtifactCompatibilityError as error:
            assert "model revision differs" in str(error)
        else:
            raise AssertionError("revision mismatch must fail closed")


def test_capture_fails_closed_inside_active_steering_session():
    wrapper = _wrapper()
    artifact = _artifact(wrapper)
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=AllTokens(),
        strength=Constant(5),
    )
    request = CaptureRequest(
        inputs=["kind words"],
        site=resid_post(0),
        positions=LastNonPaddingToken(),
    )

    with wrapper.steer(control), pytest.raises(HookLifecycleError, match="cache key"):
        wrapper.capture(request)


def test_hf_capture_supplies_offsets_for_text_span_selector():
    wrapper = _wrapper()
    result = wrapper.capture(
        CaptureRequest(
            inputs=["kind words", "cruel words"],
            site=resid_post(0),
            positions=TextSpan("words"),
        )
    )

    assert result.activations.shape == (2, 1, wrapper.hidden_size)
    assert result.attention_mask.tolist() == [[True], [True]]
