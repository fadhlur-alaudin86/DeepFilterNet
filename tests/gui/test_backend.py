"""Backend tests for DeepFilterNet GUI."""

import importlib
import os

import numpy as np

from gui.core.backend import InProcessBackend, JobConfig


def _audio() -> np.ndarray:
    """Fresh deterministic (2, 48000) float32 input chunk."""
    return np.random.default_rng(42).standard_normal((2, 48000)).astype(np.float32)


def _init_spy(calls: list):
    """Strict-signature init_df stand-in recording call kwargs and DEVICE env."""

    def spy(
        model_base_dir=None,
        post_filter=False,
        log_level="INFO",
        log_file=None,
        epoch="best",
        mask_only=False,
    ):
        calls.append(
            {
                "model_base_dir": model_base_dir,
                "post_filter": post_filter,
                "log_level": log_level,
                "log_file": log_file,
                "epoch": epoch,
                "mask_only": mask_only,
                "DEVICE": os.environ.get("DEVICE"),
            }
        )
        return object(), object(), "suffix", "best"

    return spy


def _enhance_spy(calls: list):
    """Strict-signature enhance stand-in recording call kwargs."""

    def spy(model, df_state, audio, pad=True, atten_lim_db=None):
        calls.append({"pad": pad, "atten_lim_db": atten_lim_db, "shape": tuple(audio.shape)})
        return audio.clone()

    return spy


def _fake_init_df(
    model_base_dir=None,
    post_filter=False,
    log_level="INFO",
    log_file=None,
    epoch="best",
    mask_only=False,
):
    return object(), object(), "suffix", "best"


def _fake_enhance(model, df_state, audio, pad=True, atten_lim_db=None):
    return audio.clone()


def _patch_df(monkeypatch, init_df=None, enhance=None):
    """Patch init_df/enhance on the real source module ``df.enhance``.

    gui/core/backend.py imports them lazily inside methods (global constraint),
    so the patch must live on df.enhance itself: function-level imports
    re-resolve the attribute on every call. The dotted-string form
    ``monkeypatch.setattr("df.enhance.init_df", ...)`` cannot be used because
    the package attribute ``df.enhance`` is shadowed by the ``enhance``
    function re-exported in df/__init__.py, which breaks pytest's attribute
    walk; hence the explicit module lookup via importlib.
    """
    df_enhance = importlib.import_module("df.enhance")
    monkeypatch.setattr(df_enhance, "init_df", init_df or _fake_init_df)
    monkeypatch.setattr(df_enhance, "enhance", enhance or _fake_enhance)


def test_first_enhance_initializes_once(monkeypatch):
    """2 chunks same cfg -> init_df called 1x, enhance 2x."""
    init_calls: list = []
    enhance_calls: list = []
    _patch_df(monkeypatch, _init_spy(init_calls), _enhance_spy(enhance_calls))

    backend = InProcessBackend()
    cfg = JobConfig()

    backend.enhance_chunk(_audio(), cfg)
    backend.enhance_chunk(_audio(), cfg)

    assert len(init_calls) == 1, f"Expected 1 init_df call, got {len(init_calls)}"
    assert len(enhance_calls) == 2, f"Expected 2 enhance calls, got {len(enhance_calls)}"


def test_cache_invalidated_on_param_change(monkeypatch):
    """change epoch / post_filter / device / model -> new init_df call."""
    init_calls: list = []
    _patch_df(monkeypatch, _init_spy(init_calls))
    monkeypatch.delenv("DEVICE", raising=False)

    backend = InProcessBackend()
    audio = _audio()

    backend.enhance_chunk(audio, JobConfig())
    assert len(init_calls) == 1, f"Expected 1 init_df call, got {len(init_calls)}"

    for cfg in (
        JobConfig(epoch="latest"),
        JobConfig(post_filter=False),
        JobConfig(device="CUDA:0"),
        JobConfig(model="DeepFilterNet2"),
    ):
        backend.enhance_chunk(audio, cfg)

    assert len(init_calls) == 5, (
        f"Expected a new init_df call per changed key param (5 total), " f"got {len(init_calls)}"
    )


def test_device_applied_before_init_df(monkeypatch):
    """cfg.device='CUDA:1' -> inside fake init_df spy, os.environ['DEVICE']=='cuda:1'."""
    init_calls: list = []
    _patch_df(monkeypatch, _init_spy(init_calls))
    monkeypatch.delenv("DEVICE", raising=False)

    backend = InProcessBackend()
    backend.enhance_chunk(_audio(), JobConfig(device="CUDA:1"))

    assert (
        init_calls[0]["DEVICE"] == "cuda:1"
    ), f"Expected DEVICE=cuda:1 inside init_df, got {init_calls[0]['DEVICE']!r}"


def test_auto_device_clears_env(monkeypatch):
    """cfg.device='Auto' -> spy sees 'DEVICE' absent."""
    init_calls: list = []
    _patch_df(monkeypatch, _init_spy(init_calls))
    monkeypatch.setenv("DEVICE", "cuda:0")

    backend = InProcessBackend()
    backend.enhance_chunk(_audio(), JobConfig(device="Auto"))

    assert (
        init_calls[0]["DEVICE"] is None
    ), f"Expected DEVICE absent inside init_df, got {init_calls[0]['DEVICE']!r}"


def test_pad_and_atten_mapped(monkeypatch):
    """cfg.delay_compensation=False -> enhance called with pad=False; atten_lim_db=12 -> atten_lim_db=12."""
    enhance_calls: list = []
    _patch_df(monkeypatch, enhance=_enhance_spy(enhance_calls))

    backend = InProcessBackend()
    backend.enhance_chunk(_audio(), JobConfig(delay_compensation=False, atten_lim_db=12))

    assert len(enhance_calls) == 1, f"Expected 1 enhance call, got {len(enhance_calls)}"
    assert enhance_calls[0]["pad"] is False, f"Expected pad=False, got {enhance_calls[0]['pad']!r}"
    assert (
        enhance_calls[0]["atten_lim_db"] == 12
    ), f"Expected atten_lim_db=12, got {enhance_calls[0]['atten_lim_db']!r}"


def test_no_df_stage_maps_to_mask_only(monkeypatch):
    """cfg.no_df_stage=True -> init_df kwarg mask_only=True."""
    init_calls: list = []
    _patch_df(monkeypatch, _init_spy(init_calls))

    backend = InProcessBackend()
    backend.enhance_chunk(_audio(), JobConfig(no_df_stage=True))

    assert (
        init_calls[0]["mask_only"] is True
    ), f"Expected mask_only=True, got {init_calls[0]['mask_only']!r}"


def test_enhance_chunk_shape_preserved(monkeypatch):
    """input (2, 48000) float32 -> output same shape, float32, finite."""

    def enhance_float64(model, df_state, audio, pad=True, atten_lim_db=None):
        # Return float64 on purpose to exercise the backend's float32 conversion.
        return audio.double()

    _patch_df(monkeypatch, enhance=enhance_float64)

    backend = InProcessBackend()
    result = backend.enhance_chunk(_audio(), JobConfig())

    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert result.shape == (2, 48000), f"Expected shape (2, 48000), got {result.shape}"
    assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"
    assert np.isfinite(result).all(), "Expected finite values"
    assert result.flags["C_CONTIGUOUS"], "Expected C-contiguous output"


def test_cancel_is_noop_v1(monkeypatch):
    """cancel() then enhance_chunk still works (boundary cancel owned by JobQueue, spec 3.3)."""
    enhance_calls: list = []
    _patch_df(monkeypatch, enhance=_enhance_spy(enhance_calls))

    backend = InProcessBackend()
    cfg = JobConfig()
    audio = _audio()

    backend.cancel()
    result = backend.enhance_chunk(audio, cfg)

    assert (
        len(enhance_calls) == 1
    ), f"enhance_chunk must still run after cancel(), got {len(enhance_calls)} calls"
    assert result.shape == (2, 48000), f"Expected shape (2, 48000), got {result.shape}"
