from dataclasses import replace

import pytest
import torch
import transformers

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import GenerationPhaseError, Intervention, PlanCompilationError
from repsteer.models import from_model
from repsteer.operators import Add
from repsteer.positions import GeneratedTokens
from repsteer.schedules import Constant
from repsteer.sites import resid_post


def _wrapper():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return from_model(
        transformers.LlamaForCausalLM(config).eval(),
        model_id="tiny/gated",
        revision="r1",
    )


def _artifact(wrapper):
    return DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=wrapper.revision,
            architecture=wrapper.architecture,
            site=resid_post(1),
            method="unit",
        ),
        torch.ones(wrapper.hidden_size),
    )


class _CountingSequenceGate:
    def __init__(self):
        self.evaluate_at = resid_post(0)
        self.calls = []

    def evaluate(self, context):
        activation = context.metadata["activation"]
        self.calls.append((context.phase, tuple(activation.shape)))
        return torch.ones(
            activation.shape[0], dtype=torch.bool, device=activation.device
        )

    def to_dict(self):
        return {
            "type": "counting_sequence",
            "evaluate_at": self.evaluate_at.to_dict(),
        }


def test_sequence_gate_evaluates_once_in_prefill_and_reuses_for_decode():
    wrapper = _wrapper()
    gate = _CountingSequenceGate()
    control = Intervention(
        artifact=_artifact(wrapper),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0.1),
        gate=gate,
        phase="decode",
    )
    compiled = wrapper.compile(control)

    explanation = compiled.to_dict()
    assert explanation["hook_count"] == 2
    assert (
        explanation["interventions"][0]["gate_dependency"]["cache"]
        == "prefill_sequence"
    )

    with wrapper.steer(compiled):
        wrapper.generate(
            torch.tensor([[1, 4, 5]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    assert gate.calls == [("prefill", (1, 3, wrapper.hidden_size))]


def test_sequence_gate_cache_is_reset_for_each_generation():
    wrapper = _wrapper()
    gate = _CountingSequenceGate()
    control = Intervention(
        artifact=_artifact(wrapper),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0.1),
        gate=gate,
        phase="decode",
    )

    with wrapper.steer(control):
        for token in (4, 6):
            wrapper.generate(
                torch.tensor([[1, token]]),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                min_new_tokens=2,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
            )

    assert [phase for phase, _ in gate.calls] == ["prefill", "prefill"]


def test_static_zero_strength_removes_target_and_gate_hooks_without_rng_effect():
    wrapper = _wrapper()
    gate = _CountingSequenceGate()
    control = Intervention(
        artifact=_artifact(wrapper),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0),
        gate=gate,
        phase="decode",
    )
    compiled = wrapper.compile(control)
    inputs = torch.tensor([[1, 4, 5]])
    options = {
        "attention_mask": torch.ones_like(inputs),
        "min_new_tokens": 4,
        "max_new_tokens": 4,
        "do_sample": True,
        "seed": 123,
        "use_cache": True,
    }

    baseline = wrapper.generate(inputs, **options)
    assert compiled.groups() == ()
    assert compiled.gate_groups() == ()
    assert compiled.to_dict()["hook_count"] == 0
    with wrapper.steer(compiled):
        steered = wrapper.generate(inputs, **options)

    assert gate.calls == []
    assert torch.equal(steered.token_ids, baseline.token_ids)


def test_first_decode_of_new_generation_cannot_reuse_previous_gate_cache():
    wrapper = _wrapper()
    gate = _CountingSequenceGate()
    control = Intervention(
        artifact=_artifact(wrapper),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0.1),
        gate=gate,
        phase="decode",
    )
    with torch.inference_mode():
        primed = wrapper.model(
            input_ids=torch.tensor([[1, 7]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            use_cache=True,
        )
    decode_ids = torch.tensor([[8]])
    decode_mask = torch.ones(1, 3, dtype=torch.long)

    with wrapper.steer(control):
        wrapper.generate(
            torch.tensor([[1, 4]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            min_new_tokens=2,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )
        assert len(gate.calls) == 1
        with (
            pytest.raises(GenerationPhaseError, match="no cached prefill value"),
            wrapper.generation_tracker.generation(
                input_ids=decode_ids,
                inputs_embeds=None,
                attention_mask=decode_mask,
            ),
        ):
            wrapper.model(
                input_ids=decode_ids,
                attention_mask=decode_mask,
                past_key_values=primed.past_key_values,
                use_cache=True,
            )

    assert len(gate.calls) == 1


def test_orphaned_compiled_plan_cannot_bind_to_an_identity_equivalent_wrapper():
    wrapper = _wrapper()
    control = Intervention(
        artifact=_artifact(wrapper),
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0.1),
        phase="decode",
    )
    orphaned = replace(wrapper.compile(control), _model_ref=None)
    other = _wrapper()

    with pytest.raises(PlanCompilationError, match="lost its source"):
        other.compile(orphaned)
