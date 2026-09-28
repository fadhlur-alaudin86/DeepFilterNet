"""Compatibility tests for df.io with torchaudio 2.11."""

import numpy as np
import soundfile as sf
import torch

from df.io import AudioMetaData, get_resample_params, load_audio


def test_df_io_importable():
    """Import df.io and df.enhance should not raise."""
    import df  # noqa: F401

    # Verify AudioMetaData is accessible from df.io
    assert AudioMetaData is not None


def test_audio_roundtrip(tmp_path):
    """Build 0.25 s stereo float32 noise at 48000 Hz; save and load; verify roundtrip."""
    rng = np.random.RandomState(42)
    duration = 0.25
    sr = 48000
    n_samples = int(duration * sr)
    audio_np = rng.randn(2, n_samples).astype(np.float32)
    # write stereo float32 wav with FLOAT subtype for exact roundtrip
    out_file = tmp_path / "test_roundtrip.wav"
    sf.write(str(out_file), audio_np.T, sr, subtype="FLOAT")
    loaded, info = load_audio(str(out_file))
    assert info.sample_rate == 48000, f"Expected 48000, got {info.sample_rate}"
    assert loaded.shape == audio_np.shape, f"Shape mismatch: {loaded.shape} vs {audio_np.shape}"
    assert torch.allclose(loaded, torch.from_numpy(audio_np), atol=1e-3)


def test_load_audio_resamples(tmp_path):
    """Save at 44100, load at 48000 -> returned audio length within 2% of round(0.25 * 48000)."""
    rng = np.random.RandomState(123)
    duration = 0.25
    sr_orig = 44100
    target_sr = 48000
    n_samples_orig = int(duration * sr_orig)
    audio_np = rng.randn(2, n_samples_orig).astype(np.float32)
    out_file = tmp_path / "test_resample.wav"
    sf.write(str(out_file), audio_np.T, sr_orig, subtype="FLOAT")
    loaded, info = load_audio(str(out_file), sr=target_sr)
    expected_len = round(duration * target_sr)
    # allow ~2% tolerance on length
    lo = expected_len * 0.98
    hi = expected_len * 1.02
    assert lo <= loaded.shape[-1] <= hi, f"Length {loaded.shape[-1]} not in [{lo}, {hi}]"


def test_get_resample_params_version_correct():
    """get_resample_params should return dicts without raising; sinc value should be modern constant."""
    params_fast = get_resample_params("sinc_fast")
    params_best = get_resample_params("sinc_best")
    # On torchaudio >= 2.11, sinc_interp_hann is the modern constant
    assert params_fast["resampling_method"] == "sinc_interp_hann"
    assert params_best["resampling_method"] == "sinc_interp_hann"


def test_audio_meta_data_fields():
    """AudioMetaData should have the expected fields."""
    import df.io

    # Instantiate with known values
    meta = df.io.AudioMetaData(
        sample_rate=48000, num_frames=16000, num_channels=2, bits_per_sample=32, encoding="float"
    )
    assert meta.sample_rate == 48000
    assert meta.num_frames == 16000
    assert meta.num_channels == 2
    assert meta.bits_per_sample == 32
    assert meta.encoding == "float"
