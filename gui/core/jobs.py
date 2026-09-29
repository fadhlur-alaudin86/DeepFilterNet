"""Job queue and chunked output pipeline for DeepFilterNet GUI.

Single worker thread processes queued jobs FIFO. Pause/cancel are checked
before each chunk so state transitions happen at chunk boundaries; completed
chunk parts are kept on the queue so a paused job resumes without repeating
work. All ``df``/``torch`` imports are lazy, inside the worker code paths.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from gui.core.backend import EnhancementBackend, JobConfig
from gui.core.chunker import concat_chunks, plan_chunks
from gui.core.events import AppEvent, EventBus

_WORKER_POLL_S = 0.1
_FORMATS_CACHE: list[str] | None = None


class JobState(str, Enum):
    """Lifecycle states of a queued job."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """One enhancement job: an input file list plus its immutable config snapshot."""

    id: str
    files: list[Path]
    cfg: JobConfig
    state: JobState = JobState.QUEUED
    file_results: dict[str, str] = field(default_factory=dict)  # path str -> per-file status
    progress: float = 0.0  # 0..1 across all files/chunks, monotonically nondecreasing
    error: str | None = None  # job-level message when state == FAILED
    file_index: int = 0  # cursor: next file to process (persists across pause/resume)
    chunk_index: int = 0  # cursor: next chunk within the current file


def build_output_path(src: Path, cfg: JobConfig) -> Path:
    """Return the first free output path honouring ``cfg.suffix_enabled``.

    With the suffix enabled (default): ``<stem>-deep-filtered[-<n>].<fmt>``;
    with it disabled: ``<stem>[-<n>].<fmt>``. ``<n>`` starts at 2 on collision.
    """
    out_dir = Path(cfg.output_dir)
    base = f"{src.stem}-deep-filtered" if cfg.suffix_enabled else src.stem
    candidate = out_dir / f"{base}.{cfg.output_format}"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{base}-{n}.{cfg.output_format}"
        n += 1
    return candidate


def _probe_save_formats() -> list[str]:
    """Probe the real writer: a 1-frame df.io.save_audio per candidate format.

    Returns ``[]`` when ``df.io`` cannot be imported (e.g. torch-less frozen
    bundles, where importing ``df`` fails), so view construction degrades to
    an empty format list instead of crashing (R17).
    """
    import os
    import tempfile

    import numpy as np

    try:
        from df.io import save_audio
    except ImportError:
        return []

    frame = np.ones((1, 1), dtype=np.float32)
    supported = []
    for fmt in ("wav", "flac", "mp3"):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                save_audio(os.path.join(tmp, f"probe.{fmt}"), frame, sr=48000)
            supported.append(fmt)
        except Exception:
            continue
    return supported


def supported_formats() -> list[str]:
    """Output formats that ``df.io.save_audio`` can actually write.

    A 1-frame file per candidate is probed through the real writer once per
    process; the result is cached for submit() and UI calls. Empty list when
    ``df.io`` is unimportable (degraded/torch-less environments).
    """
    global _FORMATS_CACHE
    if _FORMATS_CACHE is None:
        _FORMATS_CACHE = _probe_save_formats()
    return list(_FORMATS_CACHE)


def _model_sr() -> int:
    """Model sample rate from ``df.model.ModelParams`` (lazy import)."""
    from df.config import config
    from df.model import ModelParams

    try:
        return int(ModelParams().sr)
    except ValueError:
        # No df config loaded yet: the backend loads the model config lazily
        # on the first chunk, so fall back to df's built-in defaults.
        config.use_defaults()
        return int(ModelParams().sr)


def _failure_message(exc: Exception) -> str:
    """Per-file failure message; OOM errors get an actionable hint."""
    message = str(exc) or type(exc).__name__
    if isinstance(exc, RuntimeError) and "out of memory" in message.lower():
        return f"{message}. Reduce chunk size or switch device to CPU"
    return message


def _resample_array(audio, from_sr: int, to_sr: int):
    """Resample a numpy audio array back to the source sample rate (lazy imports)."""
    import torch

    from df.io import resample

    out = resample(torch.from_numpy(audio), from_sr, to_sr)
    return out.numpy()


class JobQueue:
    """FIFO job queue with chunk-boundary pause/cancel and a single worker thread."""

    def __init__(self, backend: EnhancementBackend, bus: EventBus) -> None:
        self._backend = backend
        self._bus = bus
        self._lock = threading.RLock()
        self._jobs: list[Job] = []
        self._pause_events: dict[str, threading.Event] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._partials: dict[str, list] = {}  # job id -> enhanced chunks of the paused file
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopping = False

    # ------------------------------------------------------------------ public

    @property
    def jobs(self) -> list[Job]:
        """Snapshot of all jobs in queue order (for UI listing and tests)."""
        with self._lock:
            return list(self._jobs)

    def submit(self, files: list[Path], cfg: JobConfig) -> Job:
        """Validate the output format, then enqueue a new job (state QUEUED)."""
        formats = supported_formats()
        if cfg.output_format not in formats:
            raise ValueError(
                f"Unsupported output format {cfg.output_format!r}; "
                f"supported formats: {', '.join(formats)}"
            )
        paths = [Path(f) for f in files]
        job = Job(
            id=uuid.uuid4().hex,
            files=paths,
            cfg=cfg,
            file_results={str(p): "pending" for p in paths},
        )
        with self._lock:
            self._jobs.append(job)
            self._pause_events[job.id] = threading.Event()
            self._cancel_events[job.id] = threading.Event()
        self._publish_state(job)
        self._wakeup.set()
        return job

    def pause(self, job_id: str) -> None:
        """Request a pause; a running job stops at the next chunk boundary."""
        with self._lock:
            job = self._find(job_id)
            if job is None or job.state in (
                JobState.PAUSED,
                JobState.DONE,
                JobState.FAILED,
                JobState.CANCELLED,
            ):
                return
            self._pause_events[job_id].set()
            direct = job.state == JobState.QUEUED
            if direct:
                job.state = JobState.PAUSED
        if direct:
            self._publish_state(job)

    def resume(self, job_id: str) -> None:
        """Requeue a paused job; drops a pause still pending on a running job."""
        with self._lock:
            job = self._find(job_id)
            if job is None:
                return
            self._pause_events[job_id].clear()
            if job.state == JobState.PAUSED:
                job.state = JobState.QUEUED
                transitioned = True
            elif job.state == JobState.RUNNING:
                # Pause requested but no chunk boundary reached yet: cancel it.
                transitioned = False
            else:
                return
        if transitioned:
            self._publish_state(job)
        self._wakeup.set()

    def cancel(self, job_id: str) -> None:
        """Cancel a job; a running job stops at the next chunk boundary."""
        with self._lock:
            job = self._find(job_id)
            if job is None or job.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
                return
            self._cancel_events[job_id].set()
            direct = job.state in (JobState.QUEUED, JobState.PAUSED)
            if direct:
                self._mark_cancelled_locked(job)
        if direct:
            self._publish_state(job)
        self._wakeup.set()

    def retry(self, job_id: str) -> None:
        """Requeue a failed job with the same snapshot, skipping already-done files."""
        with self._lock:
            job = self._find(job_id)
            if job is None or job.state != JobState.FAILED:
                return
            for path in job.files:
                key = str(path)
                if job.file_results.get(key) != "done":
                    job.file_results[key] = "pending"
            job.file_index = 0
            job.chunk_index = 0
            job.error = None
            job.state = JobState.QUEUED
            self._partials.pop(job.id, None)
            self._pause_events[job.id].clear()
            self._cancel_events[job.id].clear()
        self._publish_state(job)
        self._wakeup.set()

    def start(self) -> None:
        """Spawn the worker thread (no-op if it is already running)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._worker_loop, name="jobqueue-worker", daemon=True
            )
            thread = self._thread
        thread.start()

    def shutdown(self) -> None:
        """Cancel all pending work, join the worker, then shut down the backend."""
        with self._lock:
            self._stopping = True
            cancelled: list[Job] = []
            for job in self._jobs:
                if job.state in (JobState.DONE, JobState.FAILED, JobState.CANCELLED):
                    continue
                self._cancel_events[job.id].set()
                if job.state in (JobState.QUEUED, JobState.PAUSED):
                    self._mark_cancelled_locked(job)
                    cancelled.append(job)
            thread = self._thread
        for job in cancelled:
            self._publish_state(job)
        self._wakeup.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        self._backend.shutdown()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no job is RUNNING or QUEUED; False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                busy = any(job.state in (JobState.QUEUED, JobState.RUNNING) for job in self._jobs)
            if not busy:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def move_up(self, job_id: str) -> None:
        """Move a job one position earlier in the queue (no-op at the top)."""
        with self._lock:
            idx = self._index_of(job_id)
            if idx is None or idx == 0:
                return
            self._jobs[idx - 1], self._jobs[idx] = self._jobs[idx], self._jobs[idx - 1]
        self._wakeup.set()

    def move_down(self, job_id: str) -> None:
        """Move a job one position later in the queue (no-op at the bottom)."""
        with self._lock:
            idx = self._index_of(job_id)
            if idx is None or idx >= len(self._jobs) - 1:
                return
            self._jobs[idx + 1], self._jobs[idx] = self._jobs[idx], self._jobs[idx + 1]
        self._wakeup.set()

    # ----------------------------------------------------------------- worker

    def _worker_loop(self) -> None:
        while True:
            self._wakeup.wait(timeout=_WORKER_POLL_S)
            self._wakeup.clear()
            if self._stopping:
                break
            job = self._next_queued()
            if job is None:
                continue
            try:
                self._run_job(job)
            except Exception as exc:  # defensive: never let the worker die
                self._fail_job(job, f"unexpected error: {exc}")

    def _next_queued(self) -> Job | None:
        with self._lock:
            for job in self._jobs:
                if job.state == JobState.QUEUED:
                    job.state = JobState.RUNNING
                    chosen = job
                    break
            else:
                return None
        self._publish_state(chosen)
        return chosen

    def _run_job(self, job: Job) -> None:
        total = len(job.files)
        while True:
            if self._cancel_events[job.id].is_set():
                self._finalize_cancelled(job)
                return
            if self._pause_events[job.id].is_set():
                self._finalize_paused(job)
                return
            with self._lock:
                # Skip files already completed (retry requeues from the start).
                while (
                    job.file_index < total
                    and job.file_results.get(str(job.files[job.file_index])) == "done"
                ):
                    job.file_index += 1
                    job.chunk_index = 0
                if job.file_index >= total:
                    break
                path = job.files[job.file_index]

            status, error = self._process_file(job, path)
            if status == "paused":
                if not (
                    self._pause_events[job.id].is_set() or self._cancel_events[job.id].is_set()
                ):
                    # resume() dropped the pending pause while finalizing: keep going.
                    continue
                # Keep partial parts so resume continues without repeating chunks;
                # _finalize_paused re-checks cancel and may finalize CANCELLED instead.
                self._finalize_paused(job)
                return
            if status == "cancelled":
                self._partials.pop(job.id, None)
                self._finalize_cancelled(job)
                return
            with self._lock:
                key = str(path)
                job.file_results[key] = "done" if status == "done" else f"failed: {error}"
                job.file_index += 1
                job.chunk_index = 0
                done = sum(1 for v in job.file_results.values() if v == "done")
                file_progress = done / total if total else 1.0
            self._partials.pop(job.id, None)
            self._publish_progress(job, file_progress)

        # All files processed: finalize the job.
        if self._cancel_events[job.id].is_set():
            self._finalize_cancelled(job)
            return
        failed = [k for k, v in job.file_results.items() if v.startswith("failed:")]
        if failed:
            with self._lock:
                job.state = JobState.FAILED
                job.error = f"{len(failed)} of {total} files failed"
            self._publish_state(job)
            return
        self._publish_progress(job, 1.0)
        with self._lock:
            job.state = JobState.DONE
        self._publish_state(job)

    def _process_file(self, job: Job, path: Path) -> tuple[str, str | None]:
        """Enhance one file chunk by chunk.

        Returns ("done", None), ("failed", msg), ("paused", None) or
        ("cancelled", None). Pause/cancel are checked before every chunk call.
        """
        try:
            df_sr = _model_sr()
            from df.io import load_audio, save_audio

            audio, info = load_audio(str(path), sr=df_sr)
            orig_sr = int(info.sample_rate)
            wav = audio.numpy()
            chunks = plan_chunks(wav.shape[-1], df_sr, job.cfg.chunk_mode, job.cfg.chunk_size_s)
            parts = self._partials.setdefault(job.id, [])
            n_chunks = len(chunks)
            index = job.chunk_index
            while index < n_chunks:
                if self._cancel_events[job.id].is_set():
                    return "cancelled", None
                if self._pause_events[job.id].is_set():
                    return "paused", None
                start, end = chunks[index]
                parts.append(self._backend.enhance_chunk(wav[:, start:end], job.cfg))
                index += 1
                with self._lock:
                    job.chunk_index = index
                self._publish_progress(job, self._chunk_progress(job, index, n_chunks))
            if self._cancel_events[job.id].is_set():
                return "cancelled", None

            out = concat_chunks(parts)
            if orig_sr != df_sr:
                out = _resample_array(out, df_sr, orig_sr)
            out_path = build_output_path(path, job.cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_audio(str(out_path), out, sr=orig_sr)
            return "done", None
        except Exception as exc:  # per-file isolation: job continues with next file
            return "failed", _failure_message(exc)

    # ----------------------------------------------------------------- helpers

    def _chunk_progress(self, job: Job, index: int, n_chunks: int) -> float:
        with self._lock:
            done = sum(1 for v in job.file_results.values() if v == "done")
        return (done + index / n_chunks) / len(job.files)

    def _publish_progress(self, job: Job, value: float) -> None:
        with self._lock:
            # Clamp to keep job_progress events monotonically nondecreasing.
            job.progress = max(job.progress, min(value, 1.0))
            payload = {"job_id": job.id, "progress": job.progress}
        self._bus.publish(AppEvent("job_progress", payload))

    def _publish_state(self, job: Job) -> None:
        with self._lock:
            payload = {"job_id": job.id, "state": job.state.value, "error": job.error}
        self._bus.publish(AppEvent("job_state", payload))

    def _finalize_paused(self, job: Job) -> None:
        with self._lock:
            if self._cancel_events[job.id].is_set():
                # Cancel arrived while returning from _process_file: honor it.
                self._mark_cancelled_locked(job)
            else:
                job.state = JobState.PAUSED
        self._publish_state(job)

    def _finalize_cancelled(self, job: Job) -> None:
        with self._lock:
            self._mark_cancelled_locked(job)
        self._publish_state(job)

    def _mark_cancelled_locked(self, job: Job) -> None:
        """Set CANCELLED and discard partial results (lock must be held)."""
        job.state = JobState.CANCELLED
        for key, value in job.file_results.items():
            if value != "done":
                job.file_results[key] = "cancelled"
        self._partials.pop(job.id, None)

    def _fail_job(self, job: Job, error: str) -> None:
        with self._lock:
            job.state = JobState.FAILED
            job.error = error
        self._publish_state(job)

    def _find(self, job_id: str) -> Job | None:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    def _index_of(self, job_id: str) -> int | None:
        for index, job in enumerate(self._jobs):
            if job.id == job_id:
                return index
        return None
