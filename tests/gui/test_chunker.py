"""Chunker tests for DeepFilterNet GUI."""
import numpy as np
from gui.core.chunker import plan_chunks, concat_chunks, validate_chunk_size

SR = 48000


def test_auto_short_file_single_chunk():
    # 60s -> [(0, 60*SR)] regardless of size_s
    n_samples = 60 * SR
    chunks = plan_chunks(n_samples, SR, mode="auto")
    assert chunks == [(0, n_samples)]


def test_auto_long_file_splits_at_60s():
    # 130s, mode="auto" -> [(0,3600*SR)...] -> 3 chunks, last (120s,130s)
    n_samples = 130 * SR
    chunks = plan_chunks(n_samples, SR, mode="auto")
    assert len(chunks) == 3
    assert chunks[0] == (0, 60 * SR)
    assert chunks[1] == (60 * SR, 120 * SR)
    assert chunks[2] == (120 * SR, n_samples)


def test_preset_size_used():
    # mode="preset", size_s=30, 70s -> 3 chunks (30,30,10)
    n_samples = 70 * SR
    chunks = plan_chunks(n_samples, SR, mode="preset", size_s=30)
    assert len(chunks) == 3
    assert chunks[0] == (0, 30 * SR)
    assert chunks[1] == (30 * SR, 60 * SR)
    assert chunks[2] == (60 * SR, n_samples)


def test_custom_size_validation():
    # validate_chunk_size(4) and (601) raise ValueError; 5 and 600 pass
    import pytest
    with pytest.raises(ValueError):
        validate_chunk_size(4)
    with pytest.raises(ValueError):
        validate_chunk_size(601)
    validate_chunk_size(5)  # no exception
    validate_chunk_size(600)  # no exception


def test_invalid_mode_raises():
    # mode="bogus" -> ValueError
    try:
        plan_chunks(1000, SR, mode="bogus")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_concat_chunks_roundtrip_length():
    # random arrays split by plan_chunks, each enhanced identity -> concat total == n_samples
    n_samples = 100 * SR
    chunks = plan_chunks(n_samples, SR, mode="preset", size_s=30)
    parts = [np.zeros(chunk[1] - chunk[0]) for chunk in chunks]
    # simulate identity enhancement (no change) using concat_chunks
    total = len(concat_chunks(parts))
    assert total == n_samples


def test_empty_audio_single_empty_chunk():
    # n_samples=0 -> [(0, 0)]
    chunks = plan_chunks(0, SR, mode="auto")
    assert chunks == [(0, 0)]
