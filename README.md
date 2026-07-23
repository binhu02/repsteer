# repsteer

[![PyPI](https://img.shields.io/pypi/v/repsteer.svg)](https://pypi.org/project/repsteer/)
[![License](https://img.shields.io/github/license/binhu02/repsteer.svg)](LICENSE)

> **Documentation in progress:** The Markdown documentation and full docs site are currently under development. The Markdown available here is an LLM-generated draft and may contain inaccuracies or incomplete information.

`repsteer` is a Python toolkit for representation engineering and activation steering. It lets you learn, save, and apply interventions at semantic locations—such as “the residual-stream output of layer 20”—without writing PyTorch module paths.

It targets eager Hugging Face text models and static-image VLMs. It is intended for research, prototyping, and reproducible experiments; a steering direction is not guaranteed to work for every model or task.

```text
contrastive text -> capture activations -> Learner -> Artifact
                                                   |
prompt -> Model + Intervention -> steer() -> generated result
```

## Installation

Python 3.10+ is required. Install a PyTorch build appropriate to your CPU/CUDA environment first, then choose the extras you need:

```bash
# Core tensor and artifact APIs only
python -m pip install repsteer

# Hugging Face text models
python -m pip install "repsteer[hf]"

# SAELens SAE support
python -m pip install "repsteer[sae]"

# Static-image support for Qwen2.5-VL and InternVL
python -m pip install "repsteer[vlm]"
```

From a source checkout:

```bash
python -m pip install -e ".[dev]"
python -c "import repsteer; print(repsteer.__version__)"
```

## Quickstart: learn and apply a direction

This complete example learns a `DiffMean` direction from positive/negative contrastive text, then applies it during generation.

```python
import repsteer as rs

model = rs.models.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="bfloat16",
)

data = rs.data.ContrastivePairs.from_records([
    {"positive": "I help people when they struggle.", "negative": "I hurt people when they struggle."},
    {"positive": "Kindness and patience are valuable.", "negative": "Cruelty and impatience are valuable."},
    {"positive": "Respond with empathy and care.", "negative": "Respond with contempt and hostility."},
    {"positive": "I want others to feel safe.", "negative": "I want others to feel afraid."},
])

# Read the final non-padding-token activation from each text and compute
# positive mean - negative mean.
artifact = rs.learners.DiffMean(
    site=rs.sites.resid_post(layer=20),
    positions=rs.positions.LastNonPaddingToken(),
).fit(model, data)

control = rs.Intervention(
    artifact=artifact,
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(1.5),
    phase="decode",
)

with model.steer(control):
    result = model.generate(
        "A customer says: I have waited two weeks and I am furious. Reply briefly:",
        max_new_tokens=64,
        do_sample=False,
        decode_kwargs={"skip_special_tokens": True},
    )

print(result.text)
artifact.save("artifacts/kindness")
```

This example downloads a model and therefore needs appropriate model access, network access, and memory. `resid_post(20)` is only an example layer; check the model depth and the site you want to study for each model. `result` is a `GenerationResult` with `text`, `token_ids`, and the underlying Transformers result in `raw`. For decoder-only models, `text` normally contains both the input prompt and generated text.

### The four key objects

| Object | Purpose | Value in the quickstart |
| --- | --- | --- |
| `Site` | A semantic model location to read or write activations | `resid_post(20)` |
| `Artifact` | A learned direction, subspace, probe, or SAE feature with provenance | Result of `DiffMean(...).fit(...)` |
| `Intervention` | How, where, when, and how strongly to apply an artifact | `Add + GeneratedTokens + 1.5` |
| `SteeringPlan` | An ordered collection of interventions | Optional when using one intervention |

## Common workflows

### 1. Reuse a saved artifact

An artifact bundle is a directory containing a JSON manifest, `safetensors` tensors, and SHA-256 checksums. Loading verifies integrity by default.

```python
import repsteer as rs

artifact = rs.load_artifact("artifacts/kindness")
artifact.bind(model)  # Optional: validate model, revision, site, and hidden size early.

control = rs.Intervention(
    artifact=artifact,
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.NormRelative(ratio=0.2),
    phase="decode",
)

with model.steer(control):
    result = model.generate("A customer says: I have waited two weeks and I am furious. Reply briefly:", max_new_tokens=64)
```

The default compatibility level, `"exact"`, checks model ID, an explicit revision, site, and hidden size. VLM artifacts also check processor/preprocessing information. Use `compatibility="architecture"` or `"dimension"` only for explicit research-reuse cases where weaker checking is intended.

### 2. Combine interventions and inspect the plan first

`compile()` resolves and validates a plan without installing hooks. `explain()` and `diagnostics()` help identify incorrect sites, execution order, and direction conflicts.

```python
prefill = (
    control.with_phase("prefill")
    .with_positions(rs.positions.LastPromptToken())
    .with_strength(0.5)
    .with_priority(-10)
)

plan = rs.SteeringPlan([prefill, control])
compiled = model.compile(plan)

print(compiled.explain())
print(compiled.diagnostics().effective_rank)

with model.steer(compiled):
    result = model.generate("A customer says: I have waited two weeks and I am furious. Reply briefly:", max_new_tokens=48)
```

Interventions run first by ascending `priority`, then by declaration order. Hooks exist only inside `with model.steer(...):` and are cleaned up even when generation raises an exception.

### 3. Capture activations directly

Use `capture_activations()` to inspect activations yourself or reuse a capture result:

```python
batch = rs.capture.capture_activations(
    model,
    inputs=["Kindness matters.", "I enjoy helping."],
    site=rs.sites.resid_post(20),
    positions=rs.positions.PromptTokens(),
    pooling="mean",
    batch_size=2,
)

print(batch.shape)  # [samples, hidden_size]
```

Pass `store=rs.capture.MemoryStore()` to cache results by capture request. Capture is rejected during an active steering session or `generate()` call so that intervened activations are not accidentally reused.

## Core API

### Models and generation

| API | Purpose |
| --- | --- |
| `rs.models.from_pretrained(model_id, *, revision=None, dtype=None, tokenizer=None, processor=None, **model_kwargs)` | Lazily load a Hugging Face text model or VLM; remaining arguments are passed to Transformers. |
| `rs.models.from_model(raw_model, tokenizer=None, processor=None, adapter=None, model_id=None, revision=None)` | Wrap an already-constructed PyTorch/Transformers model. Provide stable `model_id` and `revision` values for local models. |
| `model.generate(prompt_or_inputs, **kwargs)` | Hugging Face-style generation. It accepts strings, encoded mappings, or tensors; extra options include `seed` and `decode_kwargs`. VLMs accept `image=` or `images=`. |
| `model.capture(request)` | Perform one raw activation capture; normally prefer `rs.capture.capture_activations()`. |
| `model.compile(plan, compatibility="exact")` | Validate sites, artifacts, gates, and composition order; returns a `CompiledPlan`. |
| `model.steer(plan, compatibility="exact")` | Return a context manager for calling `generate()` or `forward()` with steering active. |

### Sites: choose a model location

Built-in text-model adapters support these language-stream sites:

```python
rs.sites.resid_pre(layer)
rs.sites.attn_out(layer)
rs.sites.resid_mid(layer)
rs.sites.mlp_out(layer)
rs.sites.resid_post(layer)
```

Built-in text adapters support Gemma/Gemma2, Llama, Mistral, and Qwen2. Qwen2.5-VL and InternVL additionally support:

```python
rs.sites.vision_resid(layer)
rs.sites.projector_in()
rs.sites.projector_out()
```

`head_out()`, `logits()`, and `fusion_out()` are public site constructors, but current built-in adapters do not resolve them. This support statement applies to adapters, not a claim that every public checkpoint has been tested on hardware.

### Data, capture, and learners

```python
# Common contrastive-data constructors
rs.data.ContrastivePairs.from_records(records)
rs.data.ContrastivePairs.from_pairs(positive, negative)
rs.data.ContrastivePairs.from_groups(positive, negative)
```

Every built-in learner requires `site` and `positions`. Common options are `pooling`, `normalize`, `batch_size`, and `seed`; all use `.fit(model, data)` to produce an artifact.

| Learner | Suitable input | Output |
| --- | --- | --- |
| `rs.learners.DiffMean` | Paired or independent positive/negative samples | `DirectionArtifact` |
| `rs.learners.ActAdd` / `CAA` | Strictly aligned positive/negative pairs | `DirectionArtifact` |
| `rs.learners.PCA` | Positive/negative activation sets | `SubspaceArtifact` |
| `rs.learners.LAT` | Paired activation differences | Direction with one component; otherwise a subspace |
| `rs.learners.LinearProbe` / `LogisticProbe` | Binary positive/negative samples | `ProbeArtifact` |

`LastNonPaddingToken()` selects one activation per sample, which suits identity pooling. For multiple selected tokens, set `pooling="mean"`, `"last"`, or `"max"`:

```python
artifact = rs.learners.DiffMean(
    site=rs.sites.resid_post(20),
    positions=rs.positions.PromptTokens(),
    pooling="mean",
    batch_size=8,
).fit(model, data)
```

### Interventions, positions, and phases

```python
rs.Intervention(
    artifact=artifact,                       # Required
    operator=rs.operators.Add(),             # Required
    positions=rs.positions.GeneratedTokens(),  # Required
    strength=rs.schedules.Constant(1),       # Required
    site=None,                               # Defaults to artifact metadata's site
    gate=rs.gates.Always(),
    phase="both",                           # "prefill" | "decode" | "both"
    priority=0,
)
```

`phase` controls when a hook may run; the position selector decides which tokens or patches are changed in that phase:

| Selector | Prefill | Decode |
| --- | --- | --- |
| `PromptTokens()` | Every non-padding prompt token | Empty |
| `LastPromptToken()` | Last non-padding prompt token | Empty |
| `GeneratedTokens()` | Empty | Token in the current decode forward |
| `AllTokens()` | Every non-padding prompt token | Current token |
| `LastNonPaddingToken()` | Last valid token in the current forward | Current token |
| `TokenIndices(...)`, `TextSpan(...)`, `SpecialToken(...)` | Their respective selected positions | Depends on the current forward |
| `ImageTokens(...)`, `ImagePatches(...)`, `ObjectPatches(...)` | Corresponding VLM positions only | Empty |

Important: in ordinary cached generation, a decode intervention using `GeneratedTokens()` does not change the first generated token, because its logits come from prefill. To affect it, operate on `LastPromptToken()` (or another appropriate selector) during `prefill`. Decode steering does not rewrite the existing KV cache.

### Operators and strength schedules

| API | Behavior |
| --- | --- |
| `rs.operators.Add()` / `Subtract()` | Add/subtract a direction; accept a direction, probe weight, or rank-1 subspace. |
| `rs.operators.RemoveProjection()` | Remove the projection onto a direction or full subspace. |
| `rs.operators.Replace()` | Interpolate toward a target vector by the specified strength. |
| `rs.operators.Clamp()` / `Ablate()` | Edit an already encoded SAE latent feature. |
| `rs.operators.SAEClamp()` / `SAEAblate()` | Encode, edit, and decode residual activations through an SAE while preserving the reconstruction residual. |
| `rs.schedules.Constant(alpha)` | Use fixed strength. |
| `rs.schedules.NormRelative(ratio, p=2)` | Scale strength by the current activation norm. |

`Constant(0)` and a statically zero `NormRelative` are strict no-ops: they do not call the selector, gate, or operator, and do not install the corresponding hook. This makes them useful baselines in sweeps.

### Gates: conditional steering

| API | Purpose |
| --- | --- |
| `rs.gates.Always()` | Default gate; always enabled. |
| `rs.gates.ProbeGate(...)` | Threshold a `ProbeArtifact` prediction. |
| `rs.gates.SAEActivationGate(...)` | Threshold an SAE feature activation. |
| `rs.gates.CosineGate(...)`, `CallableGate(...)` | Custom cosine- or Python-callable-based conditions. |
| `rs.gates.AndGate(...)`, `OrGate(...)`, `NotGate(...)` | Compose gates logically. |

When an activation gate has `evaluate_at=site`, it reads a language-stream activation once in prefill and caches one decision per sample for decode. Such sequence gates cannot start from a pre-populated KV cache and cannot use a vision/projector site as `evaluate_at`.

### Artifacts and safe I/O

| API | Purpose |
| --- | --- |
| `artifact.save(path)` / `rs.artifacts.save_artifact(artifact, path)` | Save a portable artifact bundle. |
| `rs.load_artifact(path, verify_checksums=True, device="cpu")` | Verify SHA-256 checksums by default, then load. |
| `artifact.bind(model, compatibility="exact")` | Validate target-model compatibility without changing the artifact. |
| `rs.ArtifactMetadata` | Record model, site, data, and method provenance when manually constructing artifacts or selecting SAE features. |

The principal artifact types are `DirectionArtifact`, `SubspaceArtifact`, `ProbeArtifact`, and `SAEFeatureArtifact`. Artifacts store tensors and JSON metadata only; they do not load pickle or other executable payloads.

## Advanced: SAE feature steering

Install `repsteer[sae]` to select features from a SAELens SAE or your own `SAEAdapter`. This is the shortest path; replace `release` and `sae_id` with real SAE identifiers.

```python
sae = rs.sae.load(
    provider="saelens",
    release="<release>",
    sae_id="<sae-id>",
    device="cuda",
)

feature = rs.sae.select_feature(
    sae=sae,
    model=model,
    data=data,
    criterion="supervised",
    site=rs.sites.resid_post(20),
)

control = rs.Intervention(
    artifact=feature,
    operator=rs.operators.Add(),  # Add the feature's decoder direction.
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(0.5),
    phase="decode",
)
```

Use `rs.sae.select_supervised_features(...)` to get multiple features ranked by activation separation, or `rs.sae.select_causal_features(...)` to run a caller-provided causal evaluator over candidates.

Bare `Clamp` and `Ablate` apply only when the hook already supplies SAE latents. For normal residual-stream activations, pass the same runtime SAE to `SAEClamp` or `SAEAblate`:

```python
ablate = control.with_operator(rs.operators.SAEAblate(sae=sae))
clamp = control.with_operator(rs.operators.SAEClamp(value=5.0, sae=sae))
```

## Advanced: static-image VLM steering

Install `repsteer[vlm]`. Built-in VLM adapters support static images in Qwen2.5-VL and InternVL, loading an `AutoProcessor` when appropriate.

```python
from PIL import Image
import repsteer as rs

model = rs.models.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    revision="<immutable-model-commit>",
    dtype="bfloat16",
    device_map={"": "cuda:0"},
)

vision_artifact = rs.load_artifact("artifacts/vision_direction")
language_artifact = rs.load_artifact("artifacts/language_direction")

vision = rs.Intervention(
    artifact=vision_artifact,
    site=rs.sites.vision_resid(12),
    operator=rs.operators.Add(),
    positions=rs.positions.ObjectPatches(
        boxes=[(0.10, 0.10, 0.90, 0.90)],
        image_index=0,
    ),
    strength=rs.schedules.Constant(0.8),
    phase="prefill",
)
language = rs.Intervention(
    artifact=language_artifact,
    site=rs.sites.resid_post(20),
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(-0.3),
    phase="decode",
)

image = Image.open("image.jpg")
with model.steer(rs.SteeringPlan([vision, language])):
    result = model.generate(
        "Describe the main object in this image.",
        image=image,
        max_new_tokens=64,
        do_sample=False,
    )

print(result.text)
```

VLM position APIs:

| API | Selects |
| --- | --- |
| `ImageTokens(image_index=0)` | Image-placeholder tokens for that image in the language stream. |
| `ImagePatches(mask, image_index=0)` | A 2D boolean mask on the processor's original, pre-merge patch grid. |
| `ObjectPatches(boxes=[(x1, y1, x2, y2)], image_index=0)` | Patches whose centers fall in normalized `XYXY` boxes. |

For one prompt and one image, repsteer can insert the model processor's image placeholder automatically. Multi-image or prompt-batch calls must provide explicit placeholders matching image cardinality; `image=` and `images=` are mutually exclusive. Video is unsupported. For tiled InternVL inputs without original-image crop transforms, `ObjectPatches` is rejected rather than using an unreliable position mapping.

## Evaluation and search

Use `Grid` and `sweep()` for small, recorded searches over layers, strengths, and selectors:

```python
search = rs.evaluation.Grid(
    sites=[rs.sites.resid_post(layer) for layer in (18, 20)],
    strengths=[0.0, 0.5, 1.0],
    positions=[rs.positions.GeneratedTokens()],
)

report = rs.evaluation.sweep(
    model=model,
    artifact=artifact,
    data=["Reply to a customer whose order is late."],
    search=search,
    metrics=[rs.evaluation.ContainsText("sorry")],
    generate_kwargs={"max_new_tokens": 32, "do_sample": False},
)

best = report.select(maximize="contains")
report.to_json("reports/sweep.json")
```

Related APIs include `rs.evaluation.CallableMetric`, `ContainsText`, `MeanOutputLength`, and `SweepReport`. VLM experiments can use `EvaluationMetricGroups`, `MultimodalEvaluationRecord`, and `MultimodalEvaluationReport` to record representation, causal-behavior, and capability/quality measurements separately.

## Runtime boundaries

- Only eager Hugging Face runtime is supported; arbitrary custom generation loops, end-to-end beam search, and assisted/speculative decoding are not guaranteed.
- Quantized/offloaded target modules, model-parallel steering, video/audio, unlisted VLM architectures, and the VLM fusion stream are unsupported.
- Current built-in adapters do not support `head_out`, `logits`, or unit/head-level interventions.
- Dynamic recomputation of sequence gates during decode and re-entrant/checkpointed activation capture are unsupported.
- Validate artifact compatibility, direction sign, layer choice, strength, processor mapping, and task metrics for every experiment. This library does not replace safety, bias, or effectiveness evaluation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
mypy src/repsteer
ruff check .
```

The test suite uses local tiny-model configurations; CUDA tests are skipped when CUDA is unavailable.

## License

Apache-2.0. See [LICENSE](LICENSE).
