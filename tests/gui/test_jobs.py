"""Job queue and output pipeline tests for DeepFilterNet GUI."""

import time

import numpy as np
import pytest

from conftest import FakeBackend, make_wav
from gui.core import jobs as jobs_module
from gui.core.backend import JobConfig
from gui.core.events import EventBus
from gui.core.jobs import JobQueue, JobState, supported_formats


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_submit_completes_and_writes_output(tmp_path):
    """0.5s wav -> DONE, output exists, name == x-deep-filtered.wav."""
    src = make_wav(tmp_path / "x.wav", 0.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir))
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())
    queue.start()

    job = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "queue should go idle after submit"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    assert (out_dir / "x-deep-filtered.wav").exists(), "output file missing"
    names = [p.name for p in out_dir.iterdir()]
    assert names == ["x-deep-filtered.wav"], f"unexpected output names: {names}"
    assert job.file_results[str(src)] == "done", f"file result: {job.file_results}"
    assert job.progress == 1.0, f"expected progress 1.0, got {job.progress}"
    assert backend.shutdown_called, "shutdown must reach the backend"


def test_output_collision_gets_numeric_suffix(tmp_path):
    """Run twice -> second file named x-deep-filtered-2.wav."""
    src = make_wav(tmp_path / "x.wav", 0.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir))
    queue = JobQueue(FakeBackend(), EventBus())
    queue.start()

    job1 = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "first job should finish"
    job2 = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "second job should finish"
    queue.shutdown()

    assert job1.state == JobState.DONE, f"job1 state: {job1.state}"
    assert job2.state == JobState.DONE, f"job2 state: {job2.state}"
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == [
        "x-deep-filtered-2.wav",
        "x-deep-filtered.wav",
    ], f"unexpected output names: {names}"


def test_full_file_output_length_matches_input(tmp_path):
    """2.5s file, chunk 1s -> output frames == input frames."""
    import soundfile as sf

    src = make_wav(tmp_path / "long.wav", 2.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())
    queue.start()

    job = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    assert len(backend.calls) == 3, f"expected 3 chunks, got {len(backend.calls)}"
    out_file = out_dir / "long-deep-filtered.wav"
    assert out_file.exists(), "output file missing"
    in_frames = sf.info(str(src)).frames
    out_frames = sf.info(str(out_file)).frames
    assert out_frames == in_frames, f"frames {out_frames} != input {in_frames}"


def test_pause_resumes_at_next_chunk(tmp_path):
    """FakeBackend counts calls; pause during chunk 2 -> after resume total calls == n_chunks."""
    src = make_wav(tmp_path / "p.wav", 2.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())

    job = queue.submit([src], cfg)  # queued before the worker starts
    backend.on_call = lambda n: queue.pause(job.id) if n == 2 else None
    queue.start()

    assert _wait_for(lambda: job.state == JobState.PAUSED), f"state: {job.state}"
    assert len(backend.calls) == 2, f"expected pause before chunk 3, calls={len(backend.calls)}"

    queue.resume(job.id)
    assert queue.wait_idle(30.0), "queue should go idle after resume"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    assert (
        len(backend.calls) == 3
    ), f"expected n_chunks==3 calls with no chunk repeat, got {len(backend.calls)}"
    assert (out_dir / "p-deep-filtered.wav").exists(), "output file missing"


def test_cancel_discards_partial_output(tmp_path):
    """cancel mid-file -> state CANCELLED, no output file on disk."""
    src = make_wav(tmp_path / "c.wav", 2.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())

    job = queue.submit([src], cfg)
    backend.on_call = lambda n: queue.cancel(job.id) if n == 2 else None
    queue.start()

    assert _wait_for(lambda: job.state == JobState.CANCELLED), f"state: {job.state}"
    queue.shutdown()

    assert job.state == JobState.CANCELLED, f"expected CANCELLED, got {job.state}"
    assert job.file_results[str(src)] == "cancelled", f"results: {job.file_results}"
    assert not out_dir.exists() or not any(
        out_dir.iterdir()
    ), f"partial output must be discarded, found: {list(out_dir.glob('*'))}"


def test_failed_file_does_not_block_others(tmp_path):
    """[good.wav, bad.txt] -> job FAILED, good.wav result 'done', bad.txt starts with 'failed:'."""
    good = make_wav(tmp_path / "good.wav", 0.5)
    bad = tmp_path / "bad.txt"
    bad.write_text("this is not audio")
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir))
    queue = JobQueue(FakeBackend(), EventBus())
    queue.start()

    job = queue.submit([good, bad], cfg)
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.FAILED, f"expected FAILED, got {job.state}"
    assert job.error == "1 of 2 files failed", f"job error: {job.error!r}"
    assert job.file_results[str(good)] == "done", f"results: {job.file_results}"
    assert job.file_results[str(bad)].startswith("failed:"), f"results: {job.file_results}"
    assert (out_dir / "good-deep-filtered.wav").exists(), "good.wav output missing"


def test_oom_error_includes_actionable_hint(tmp_path):
    """armed OOM -> file_results msg contains 'chunk' and 'CPU'."""
    src = make_wav(tmp_path / "o.wav", 0.5)
    backend = FakeBackend()
    backend.oom_on_call = 1
    queue = JobQueue(backend, EventBus())
    queue.start()

    job = queue.submit([src], JobConfig(output_dir=str(tmp_path / "out")))
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.FAILED, f"expected FAILED, got {job.state}"
    msg = job.file_results[str(src)]
    assert msg.startswith("failed:"), f"unexpected file result: {msg!r}"
    assert "chunk" in msg, f"hint must mention 'chunk': {msg!r}"
    assert "CPU" in msg, f"hint must mention 'CPU': {msg!r}"


def test_unsupported_format_rejected_at_submit(tmp_path, monkeypatch):
    """submit(['x.wav'], output_format='mp3') without mp3 in supported_formats -> ValueError first."""
    src = make_wav(tmp_path / "x.wav", 0.5)
    monkeypatch.setattr(jobs_module, "supported_formats", lambda: ["wav", "flac"])
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())

    with pytest.raises(ValueError):
        queue.submit([src], JobConfig(output_dir=str(tmp_path / "out"), output_format="mp3"))
    queue.shutdown()

    assert backend.calls == [], f"backend must not be touched, got {len(backend.calls)} calls"


def test_progress_events_published(tmp_path):
    """bus captures job_progress events with monotonically nondecreasing payload['progress']."""
    src = make_wav(tmp_path / "g.wav", 2.5)
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    queue = JobQueue(FakeBackend(), bus)
    queue.start()

    job = queue.submit([src], JobConfig(output_dir=str(tmp_path / "out")))
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    progress = [e.payload["progress"] for e in events if e.type == "job_progress"]
    assert progress, "expected job_progress events"
    assert all(
        b >= a for a, b in zip(progress, progress[1:])
    ), f"progress must be monotonic nondecreasing: {progress}"
    assert progress[-1] == 1.0, f"final progress must be 1.0, got {progress[-1]}"


def test_move_up_down_reorders_pending_jobs(tmp_path):
    """move_up/move_down swap adjacent queued jobs; edge moves are no-ops."""
    files = [make_wav(tmp_path / f"{name}.wav", 0.1) for name in ("a", "b", "c")]
    cfg = JobConfig(output_dir=str(tmp_path / "out"))
    queue = JobQueue(FakeBackend(), EventBus())

    job_a = queue.submit([files[0]], cfg)
    job_b = queue.submit([files[1]], cfg)
    job_c = queue.submit([files[2]], cfg)
    ids = lambda: [j.id for j in queue.jobs]  # noqa: E731
    assert ids() == [job_a.id, job_b.id, job_c.id], f"initial order: {ids()}"

    queue.move_up(job_c.id)
    assert ids() == [job_a.id, job_c.id, job_b.id], f"after move_up: {ids()}"

    queue.move_down(job_a.id)
    assert ids() == [job_c.id, job_a.id, job_b.id], f"after move_down: {ids()}"

    queue.move_up(job_c.id)  # already first -> no-op
    assert ids() == [job_c.id, job_a.id, job_b.id], f"edge move_up must be a no-op: {ids()}"

    queue.move_down(job_b.id)  # already last -> no-op
    assert ids() == [job_c.id, job_a.id, job_b.id], f"edge move_down must be a no-op: {ids()}"
    queue.shutdown()


def test_supported_formats_roundtrip_via_df_io(tmp_path):
    """every supported_formats() entry must write and load back through df.io."""
    from df.io import load_audio, save_audio

    fmts = supported_formats()
    assert "wav" in fmts, f"wav must be supported, got {fmts}"
    frames = 4800
    audio = np.full((1, frames), 0.25, dtype=np.float32)
    for i, fmt in enumerate(fmts):
        probe = tmp_path / f"probe-{i}.{fmt}"
        save_audio(str(probe), audio, sr=48000)
        loaded, info = load_audio(str(probe), verbose=False)
        assert info.sample_rate == 48000, f"{fmt}: loaded sr {info.sample_rate}"
        assert loaded.shape[0] == audio.shape[0], f"{fmt}: channels {loaded.shape}"
        assert loaded.shape[-1] >= 1, f"{fmt}: nothing loaded back"


def test_pause_then_immediate_resume_never_pauses(tmp_path):
    """pause+resume inside one chunk -> no PAUSED transition, every chunk runs once."""
    src = make_wav(tmp_path / "pr.wav", 2.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    backend = FakeBackend()
    queue = JobQueue(backend, bus)

    job = queue.submit([src], cfg)

    def _pause_then_resume(n):
        if n == 1:
            queue.pause(job.id)
            queue.resume(job.id)

    backend.on_call = _pause_then_resume
    queue.start()
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    states = [e.payload["state"] for e in events if e.type == "job_state"]
    assert "paused" not in states, f"pending pause must be dropped, states: {states}"
    assert len(backend.calls) == 3, f"expected 3 chunks without repeat, calls={len(backend.calls)}"
    assert (out_dir / "pr-deep-filtered.wav").exists(), "output file missing"


def test_cancel_during_pause_finalize_ends_cancelled(tmp_path):
    """cancel arriving after _process_file returns 'paused' -> finalize CANCELLED."""
    src = make_wav(tmp_path / "cf.wav", 2.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    backend = FakeBackend()

    class _CancelDuringPauseQueue(JobQueue):
        """Test seam: set the cancel event after _process_file returns 'paused'."""

        def _process_file(self, job, path):
            status, error = super()._process_file(job, path)
            if status == "paused":
                with self._lock:
                    self._cancel_events[job.id].set()
            return status, error

    queue = _CancelDuringPauseQueue(backend, EventBus())
    job = queue.submit([src], cfg)
    backend.on_call = lambda n: queue.pause(job.id) if n == 1 else None
    queue.start()

    assert _wait_for(lambda: job.state == JobState.CANCELLED), f"state: {job.state}"
    queue.shutdown()

    assert job.state == JobState.CANCELLED, f"expected CANCELLED, got {job.state}"
    assert job.file_results[str(src)] == "cancelled", f"results: {job.file_results}"
    assert not out_dir.exists() or not any(
        out_dir.iterdir()
    ), f"partial output must be discarded, found: {list(out_dir.glob('*'))}"


def test_non_48k_input_resampled_back(tmp_path):
    """16 kHz input -> output loads at 16000 Hz, frames within +-8 of input."""
    import soundfile as sf

    sr_in = 16000
    src = make_wav(tmp_path / "s16.wav", 1.5, sr=sr_in)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), chunk_mode="preset", chunk_size_s=1)
    backend = FakeBackend()
    queue = JobQueue(backend, EventBus())
    queue.start()

    job = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "queue should go idle"
    queue.shutdown()

    assert job.state == JobState.DONE, f"expected DONE, got {job.state}"
    out_file = out_dir / "s16-deep-filtered.wav"
    assert out_file.exists(), "output file missing"
    in_info = sf.info(str(src))
    out_info = sf.info(str(out_file))
    assert out_info.samplerate == sr_in, f"expected {sr_in} Hz, got {out_info.samplerate}"
    assert (
        abs(out_info.frames - in_info.frames) <= 8
    ), f"frames {out_info.frames} vs input {in_info.frames} (tolerance 8)"


def test_suffix_disabled_keeps_original_name(tmp_path):
    """suffix_enabled=False -> y.wav first run, y-2.wav on collision."""
    src = make_wav(tmp_path / "y.wav", 0.5)
    out_dir = tmp_path / "out"
    cfg = JobConfig(output_dir=str(out_dir), suffix_enabled=False)
    queue = JobQueue(FakeBackend(), EventBus())
    queue.start()

    job1 = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "first job should finish"
    job2 = queue.submit([src], cfg)
    assert queue.wait_idle(30.0), "second job should finish"
    queue.shutdown()

    assert job1.state == JobState.DONE, f"job1 state: {job1.state}"
    assert job2.state == JobState.DONE, f"job2 state: {job2.state}"
    names = sorted(p.name for p in out_dir.iterdir())
    assert names == ["y-2.wav", "y.wav"], f"unexpected output names: {names}"


def test_move_up_reorders_pending_jobs(tmp_path):
    """move_up pulls a pending job one position earlier in queue order (Task 11)."""
    files = [make_wav(tmp_path / f"{name}.wav", 0.1) for name in ("a", "b")]
    cfg = JobConfig(output_dir=str(tmp_path / "out"))
    queue = JobQueue(FakeBackend(), EventBus())

    job_a = queue.submit([files[0]], cfg)
    job_b = queue.submit([files[1]], cfg)
    queue.move_up(job_b.id)

    assert [job.id for job in queue.jobs] == [
        job_b.id,
        job_a.id,
    ], f"move_up must reorder pending jobs: {[job.id for job in queue.jobs]}"
    queue.shutdown()
