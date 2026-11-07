"""CNN -> LSTM -> fusion over ordered crops: ordering, padding, provenance, failure handling."""
import numpy as np
import pytest

from src.pipeline.features import (
    CropSequence, FeatureExtractionError, build_untrained_upstream, run_cnn_lstm_fusion,
)


def make_seq(n, seed=0, size=112):
    rng = np.random.RandomState(seed)
    imgs = rng.randint(0, 255, (n, size, size, 3)).astype(np.uint8)
    return CropSequence(imgs, [i * 500.0 for i in range(n)], list(range(n)), [i * 5 for i in range(n)])


def make_distinct_seq(n, size=112):
    """Frames with clearly different content (brightness ramp + shifting stripes), so per-frame
    embeddings differ materially even with untrained weights."""
    imgs = np.zeros((n, size, size, 3), np.uint8)
    xs = np.arange(size)
    for i in range(n):
        stripes = ((xs // (4 + 3 * i)) % 2) * 120
        imgs[i] = np.clip(20 + 35 * i + stripes[None, :, None], 0, 255)
        imgs[i, : size // 2, :, 0] = np.clip(255 - 40 * i, 0, 255)
    return CropSequence(imgs, [i * 500.0 for i in range(n)], list(range(n)), [i * 5 for i in range(n)])


@pytest.fixture(scope="module")
def upstream(settings):
    return build_untrained_upstream(settings)


def test_untrained_weights_are_labelled_untrained(upstream, settings):
    assert not upstream.trained
    out = run_cnn_lstm_fusion(make_seq(6), upstream, settings.sequence_length)
    assert out.provenance["kind"] == "untrained_random_init"
    assert out.summary()["weights_provenance"]["kind"] == "untrained_random_init"


def test_shapes_and_counts(upstream, settings):
    out = run_cnn_lstm_fusion(make_seq(7), upstream, settings.sequence_length)
    assert out.cnn_embeddings.shape == (7, settings.cnn_feature_dim)
    assert out.lstm_repr.shape == (settings.lstm_hidden_dim,)
    assert out.fused.shape == (settings.fusion_output_dim,)
    assert (out.n_real, out.n_padded) == (7, settings.sequence_length - 7)
    assert out.timestamps_ms == [i * 500.0 for i in range(7)]


def test_extraction_is_deterministic(upstream, settings):
    a = run_cnn_lstm_fusion(make_seq(6), upstream, settings.sequence_length)
    b = run_cnn_lstm_fusion(make_seq(6), upstream, settings.sequence_length)
    assert np.array_equal(a.fused, b.fused)


def test_cnn_embedding_is_per_frame_and_lstm_sees_temporal_order(upstream, settings):
    seq = make_distinct_seq(6)
    fwd = run_cnn_lstm_fusion(seq, upstream, settings.sequence_length)
    # (an UNTRAINED CNN is nearly input-insensitive: ~4e-3 spread here; runs are bit-exact, so
    # any difference above float noise is a real effect)
    assert np.ptp(fwd.cnn_embeddings, axis=0).max() > 1e-3, "test frames must yield distinct embeddings"
    rev_seq = CropSequence(seq.images[::-1].copy(), seq.timestamps_ms, seq.sample_indices, seq.frame_indices)
    rev = run_cnn_lstm_fusion(rev_seq, upstream, settings.sequence_length)
    # per-frame CNN embeddings are just reordered ...
    assert np.allclose(fwd.cnn_embeddings, rev.cnn_embeddings[::-1], atol=1e-5)
    # ... but the temporal representation depends on the order
    assert np.abs(fwd.lstm_repr - rev.lstm_repr).max() > 1e-5
    assert np.abs(fwd.fused - rev.fused).max() > 1e-5


def test_padding_is_mask_only_and_does_not_change_the_result(upstream, settings):
    seq = make_seq(6, seed=4)
    a = run_cnn_lstm_fusion(seq, upstream, 10)
    b = run_cnn_lstm_fusion(seq, upstream, 25)
    assert (a.n_padded, b.n_padded) == (4, 19)
    assert np.allclose(a.lstm_repr, b.lstm_repr, atol=1e-5) and np.allclose(a.fused, b.fused, atol=1e-5)


def test_more_crops_than_sequence_length_are_evenly_subsampled_in_order(upstream):
    out = run_cnn_lstm_fusion(make_seq(12, seed=5), upstream, 5)
    assert out.subsampled and out.n_real == 5 and out.n_available == 12 and out.n_padded == 0
    assert out.timestamps_ms == sorted(out.timestamps_ms)
    assert out.timestamps_ms[0] == 0.0 and out.timestamps_ms[-1] == 11 * 500.0


def test_no_crops_is_an_error_not_an_empty_success(upstream, settings):
    empty = CropSequence(np.zeros((0, 0, 0, 3), np.uint8), [], [], [])
    with pytest.raises(FeatureExtractionError, match="No target crops"):
        run_cnn_lstm_fusion(empty, upstream, settings.sequence_length)


def test_non_finite_output_raises_instead_of_reporting_success(settings):
    up = build_untrained_upstream(settings)
    import torch
    with torch.no_grad():
        up.lstm.lstm.weight_ih_l0.fill_(float("nan"))
    with pytest.raises(FeatureExtractionError, match="non-finite"):
        run_cnn_lstm_fusion(make_seq(5), up, settings.sequence_length)
