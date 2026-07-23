from types import SimpleNamespace

import torch

from repsteer.artifacts import DirectionArtifact, ProbeArtifact, SubspaceArtifact
from repsteer.capture import ActivationBatch, MemoryStore
from repsteer.data import ContrastivePairs
from repsteer.learners import LAT, PCA, ActAdd, DiffMean, LinearProbe
from repsteer.positions import LastNonPaddingToken
from repsteer.sites import resid_post


class _CaptureModel:
    model_id = "tiny/capture"
    revision = "rev-1"
    config = SimpleNamespace(_commit_hash="rev-1")

    def __init__(self):
        self.calls = 0
        self.values = {
            "p1": torch.tensor([2.0, 0.0, 0.2]),
            "p2": torch.tensor([4.0, 0.2, 0.0]),
            "p3": torch.tensor([3.0, -0.1, 0.1]),
            "n1": torch.tensor([0.0, 0.0, -0.2]),
            "n2": torch.tensor([0.0, 0.2, 0.0]),
            "n3": torch.tensor([-1.0, -0.1, -0.1]),
        }

    def capture(self, request):
        self.calls += 1
        return ActivationBatch(
            torch.stack([self.values[value] for value in request.inputs]),
            request=request,
        )


def _data():
    return ContrastivePairs.from_records(
        [
            {"positive": "p1", "negative": "n1"},
            {"positive": "p2", "negative": "n2"},
            {"positive": "p3", "negative": "n3"},
        ]
    )


def _learner(learner_type, **kwargs):
    return learner_type(
        site=resid_post(0),
        positions=LastNonPaddingToken(),
        **kwargs,
    )


def test_direction_and_subspace_learners_have_deterministic_math():
    model = _CaptureModel()
    data = _data()

    diff = _learner(DiffMean).fit(model, data)
    actadd = _learner(ActAdd).fit(model, data)
    pca = _learner(PCA, n_components=2).fit(model, data)
    lat = _learner(LAT).fit(model, data)

    assert isinstance(diff, DirectionArtifact)
    assert isinstance(actadd, DirectionArtifact)
    assert isinstance(pca, SubspaceArtifact) and pca.basis.shape == (2, 3)
    assert isinstance(lat, DirectionArtifact)
    assert torch.allclose(diff.direction.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(diff.direction, actadd.direction, atol=1e-6)
    assert torch.dot(diff.direction, torch.tensor([1.0, 0.0, 0.0])) > 0.99
    assert diff.metadata.dataset_fingerprint == data.fingerprint
    assert diff.metadata.method == "diff_mean"


def test_linear_probe_separates_examples_and_capture_store_is_reused():
    model = _CaptureModel()
    data = _data()
    store = MemoryStore()
    learner = _learner(LinearProbe, max_iter=50)

    probe = learner.fit(model, data, store=store)
    calls_after_first_fit = model.calls
    repeated = learner.fit(model, data, store=store)

    assert isinstance(probe, ProbeArtifact)
    positives = torch.stack([model.values[value] for value in data.positives])
    negatives = torch.stack([model.values[value] for value in data.negatives])
    assert probe.probabilities(positives).mean() > probe.probabilities(negatives).mean()
    assert torch.equal(probe.weight, repeated.weight)
    assert model.calls == calls_after_first_fit == 2
