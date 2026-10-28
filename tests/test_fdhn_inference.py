import io
import pickle
import zipfile
import collections
from pathlib import Path

import pytest

from fdhn.inference import (
    NUMERIC_CONTEXT_FIELDS, TEXT_CONTEXT_FIELDS, fake_score_from_probabilities,
    split_checkpoint, validate_inputs,
)


class _Storage:
    def __init__(self, *metadata):
        self.metadata = metadata


class _Tensor:
    def __init__(self, storage, offset, shape, stride, *rest):
        self.shape = shape


class _SafeMetadataUnpickler(pickle.Unpickler):
    """Inspect tensor shapes with a narrow allowlist; never instantiate torch objects."""
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, shape, stride, *rest: _Tensor(storage, offset, shape, stride, *rest)
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda tensor, *rest: tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            return _Storage(*pid[1:])
        raise pickle.UnpicklingError("blocked persistent id")


def test_training_time_metadata_and_numeric_schema():
    text, counts = validate_inputs(
        "The legislature passed a bill.",
        {"speaker": "Alex", "subject": "budget"},
        dict(zip(NUMERIC_CONTEXT_FIELDS, [0, 1, 2, 3, 4])),
    )
    assert text["speaker"] == "Alex"
    assert text["job_title"] == "NaN"
    assert counts == [0, 1, 2, 3, 4]


def test_unknown_numeric_context_is_not_guessed():
    with pytest.raises(ValueError, match="is required"):
        validate_inputs("A statement", {}, {"false_counts": 0})


def test_numeric_context_rejects_fractional_counts():
    with pytest.raises(ValueError, match="integer"):
        validate_inputs("A statement", {}, dict(zip(NUMERIC_CONTEXT_FIELDS, [0, 1, 2.5, 3, 4])))


def test_six_class_truthfulness_mapping():
    assert fake_score_from_probabilities([1, 0, 0, 0, 0, 0]) == 1
    assert fake_score_from_probabilities([0, 0, 0, 1, 0, 0]) == 0.5
    assert fake_score_from_probabilities([0, 0, 0, 0, 0, 1]) == 0


def test_checkpoint_metadata_marker_is_required_for_corrected_fusion():
    state = {"weight": object()}
    legacy_state, legacy_meta = split_checkpoint(state)
    corrected_state, corrected_meta = split_checkpoint({
        "state_dict": state,
        "meta": {"lstm_batch_order_bug_fixed": True},
    })
    assert legacy_state is state
    assert legacy_meta == {}
    assert corrected_state is state
    assert corrected_meta["lstm_batch_order_bug_fixed"] is True


def test_supplied_checkpoint_has_expected_fdhn_branch_keys():
    checkpoint = Path(__file__).resolve().parents[1] / "fdhn/checkpoints/BT+TC+CB+FZ.pt"
    assert checkpoint.is_file(), "copy the supplied BT+TC+CB+FZ.pt into fdhn/checkpoints"
    with zipfile.ZipFile(checkpoint) as archive:
        record = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        data = archive.read(record)
    state = _SafeMetadataUnpickler(io.BytesIO(data)).load()
    expected_shapes = {
        "bert.bert.embeddings.word_embeddings.weight": (30522, 768),
        "textcnn2.embedding.weight": (30522, 60),
        "cnn_bilstm.rnn.weight_ih_l0": (256, 32),
        "fuzzy.membership_miu": (6, 6),
        "fuse.weight": (6, 24),
    }
    for name, shape in expected_shapes.items():
        assert state[name].shape == shape
