"""In-process enhancement backend for DeepFilterNet GUI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gui.core.device import apply_device

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class JobConfig:
    """Immutable job parameters shared by the UI, queue and backend."""

    model: str = "DeepFilterNet3"
    epoch: str = "best"
    post_filter: bool = True
    atten_lim_db: int | None = None
    device: str = "Auto"
    no_df_stage: bool = False
    delay_compensation: bool = True
    chunk_mode: str = "auto"
    chunk_size_s: int = 60
    output_dir: str = "./out"
    output_format: str = "wav"
    suffix_enabled: bool = True
    log_level: str = "INFO"


class EnhancementBackend(ABC):
    """Interface for pluggable enhancement backends."""

    @abstractmethod
    def enhance_chunk(self, audio: "np.ndarray", cfg: JobConfig) -> "np.ndarray":
        """Enhance one audio chunk under the given job configuration."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the in-flight work; chunk-boundary semantics are owned by JobQueue."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources (loaded models, caches)."""


class InProcessBackend(EnhancementBackend):
    """Enhances audio chunks in-process, keeping one loaded model per config."""

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple] = {}

    def _ensure_model(self, cfg: JobConfig) -> tuple[object, object]:
        apply_device(cfg.device)
        key = (cfg.model, cfg.epoch, cfg.post_filter, cfg.no_df_stage, cfg.device)
        if key not in self._cache:
            # Lazy import: df must not be imported at module load (global constraint).
            from df.enhance import init_df

            model, df_state, _suffix, _epoch = init_df(
                model_base_dir=cfg.model,
                post_filter=cfg.post_filter,
                log_level=cfg.log_level,
                log_file=None,
                epoch=cfg.epoch,
                mask_only=cfg.no_df_stage,
            )
            self._cache[key] = (model, df_state)
        return self._cache[key]

    def enhance_chunk(self, audio: "np.ndarray", cfg: JobConfig) -> "np.ndarray":
        # Lazy imports: torch/numpy must not load at module import (global
        # constraint) so a torch-less environment can still start degraded.
        import numpy as np
        import torch

        model, df_state = self._ensure_model(cfg)
        from df.enhance import enhance

        enhanced = enhance(
            model,
            df_state,
            torch.from_numpy(audio),
            pad=cfg.delay_compensation,
            atten_lim_db=cfg.atten_lim_db,
        )
        result = enhanced.cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(result)

    def cancel(self) -> None:
        """No-op in v1; chunk-boundary cancellation is owned by JobQueue (spec 3.3)."""

    def shutdown(self) -> None:
        self._cache.clear()
