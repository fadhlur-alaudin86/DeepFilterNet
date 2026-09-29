"""Shared test helpers promoted from the Task 7 job-pipeline tests (R9)."""

from pathlib import Path

import numpy as np

from gui.core.backend import EnhancementBackend, JobConfig

SR = 48000


class FakeBackend(EnhancementBackend):
    """Enhancement backend stand-in: audio * 0.5, records call shapes, armable OOM."""

    def __init__(self):
        self.calls: list[tuple] = []  # input chunk shapes, one per enhance_chunk call
        self.oom_on_call: int | None = None  # 1-based call number that raises
        self.on_call = None  # optional hook: fn(call_number)
        self.shutdown_called = False

    def enhance_chunk(self, audio: np.ndarray, cfg: JobConfig) -> np.ndarray:
        n = len(self.calls) + 1
        self.calls.append(tuple(audio.shape))
        if self.on_call is not None:
            self.on_call(n)
        if self.oom_on_call == n:
            raise RuntimeError("CUDA out of memory")
        return audio * 0.5

    def cancel(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdown_called = True


def make_wav(path: Path, seconds: float = 0.5, sr: int = SR) -> Path:
    """Write a tiny stereo sine wav into *path* using df.io.save_audio."""
    from df.io import save_audio

    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float64)
    audio = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
    save_audio(str(path), audio.astype(np.float32), sr=sr)
    return path


def pytest_configure(config):
    """Register the e2e marker (deselected in CI via ``-m "not e2e"``)."""
    config.addinivalue_line("markers", "e2e: runs the real model on CPU; excluded from CI runs")
