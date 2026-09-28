import torch


def test_df_io_importable():
    import df.io  # noqa: F401
    from df import enhance, init_df  # noqa: F401


def test_audio_roundtrip(tmp_path):
    from df.io import load_audio, save_audio

    sr = 48000
    duration_s = 0.25
    num_frames = int(round(duration_s * sr))
    generator = torch.Generator().manual_seed(42)
    # Uniform noise in [-0.5, 0.5], so that the PCM_16 round-trip cannot clip.
    audio = torch.rand(2, num_frames, generator=generator, dtype=torch.float32) - 0.5

    out_file = tmp_path / "roundtrip.wav"
    save_audio(str(out_file), audio, sr=sr)

    loaded, info = load_audio(str(out_file))

    assert info.sample_rate == 48000
    assert loaded.shape == audio.shape
    assert torch.allclose(loaded, audio, atol=1e-3)


def test_load_audio_resamples(tmp_path):
    from df.io import load_audio, save_audio

    orig_sr = 44100
    target_sr = 48000
    duration_s = 0.25
    num_frames = int(round(duration_s * orig_sr))
    generator = torch.Generator().manual_seed(42)
    audio = torch.rand(2, num_frames, generator=generator, dtype=torch.float32) - 0.5

    out_file = tmp_path / "resample.wav"
    save_audio(str(out_file), audio, sr=orig_sr)

    loaded, _ = load_audio(str(out_file), sr=target_sr)

    expected_len = round(0.25 * target_sr)
    actual_len = loaded.shape[-1]
    assert abs(actual_len - expected_len) <= 0.02 * expected_len


def test_get_resample_params_version_correct():
    from df.io import get_resample_params

    sinc_fast = get_resample_params("sinc_fast")
    kaiser_best = get_resample_params("kaiser_best")

    assert isinstance(sinc_fast, dict)
    assert isinstance(kaiser_best, dict)
    assert sinc_fast["resampling_method"] == "sinc_interp_hann"
