"""Queue, log and settings view tests for DeepFilterNet GUI (Task 11)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import flet as ft
import pytest

from gui.core.backend import EnhancementBackend, JobConfig
from gui.core.config import ConfigStore
from gui.core.dependency import RuntimeInfo
from gui.core.events import AppEvent, EventBus
from gui.core.jobs import Job, JobQueue, JobState
from gui.ui.pages.queue_view import QueueView
from gui.ui.pages.settings_view import SettingsView
from gui.ui.widgets.console import Console


class FakeBackend(EnhancementBackend):
    """Backend stand-in: the queue never starts its worker in these tests."""

    def enhance_chunk(self, audio, cfg):
        return audio

    def cancel(self):
        pass

    def shutdown(self):
        pass


class FakeQueue:
    """JobQueue stand-in: records per-job calls made from button handlers."""

    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.calls: list[tuple[str, str]] = []

    def pause(self, job_id):
        self.calls.append(("pause", job_id))

    def resume(self, job_id):
        self.calls.append(("resume", job_id))

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))

    def retry(self, job_id):
        self.calls.append(("retry", job_id))

    def move_up(self, job_id):
        self.calls.append(("move_up", job_id))

    def move_down(self, job_id):
        self.calls.append(("move_down", job_id))


class RecordingBus(EventBus):
    """EventBus that keeps every published event for assertions."""

    def __init__(self):
        super().__init__()
        self.events: list[AppEvent] = []

    def publish(self, event: AppEvent) -> None:
        self.events.append(event)
        super().publish(event)


@pytest.fixture
def fake_bus():
    return RecordingBus()


@pytest.fixture
def fake_queue(fake_bus):
    # Real queue so retry/pause state transitions run for real; the worker
    # thread is never started, so job states stay exactly as tests set them.
    return JobQueue(FakeBackend(), fake_bus)


@pytest.fixture
def queue_view(fake_queue, fake_bus):
    return QueueView(fake_queue, fake_bus)


@pytest.fixture
def job(fake_queue, tmp_path):
    return fake_queue.submit([tmp_path / "a.wav"], JobConfig(output_dir=str(tmp_path / "out")))


@pytest.fixture
def failed_job(fake_queue, tmp_path):
    job = fake_queue.submit([tmp_path / "bad.wav"], JobConfig(output_dir=str(tmp_path / "out")))
    job.state = JobState.FAILED
    job.error = "1 of 1 files failed"
    job.file_results = {str(tmp_path / "bad.wav"): "failed: boom"}
    return job


@pytest.fixture
def fake_queue_view():
    job = Job(
        id="job-1",
        files=[Path("a.wav")],
        cfg=JobConfig(output_dir="out"),
        state=JobState.RUNNING,
        file_results={"a.wav": "pending"},
    )
    return QueueView(FakeQueue([job]), EventBus())


@pytest.fixture
def console():
    return Console()


@pytest.fixture
def settings_view(fake_bus, tmp_path):
    return SettingsView(ConfigStore(tmp_path / "config.json"), fake_bus)


def find_controls(control, control_type):
    """Depth-first search returning every *control_type* in a control tree."""
    found = []
    if isinstance(control, control_type):
        found.append(control)
    for child in getattr(control, "controls", None) or []:
        found.extend(find_controls(child, control_type))
    content = getattr(control, "content", None)
    if isinstance(content, ft.Control):
        found.extend(find_controls(content, control_type))
    return found


def find_button(tree, label, job_id):
    """Return the FilledButton with the given label bound to *job_id*."""
    for button in find_controls(tree, ft.FilledButton):
        if button.content == label and button.data == job_id:
            return button
    raise AssertionError(f"button {label!r} for job {job_id!r} not found in tree")


def test_queue_renders_job_and_buttons(queue_view, job):
    # The view subscribed at construction: the submit event already rendered.
    initial = [chip.label.value for chip in find_controls(queue_view.cards, ft.Chip)]
    assert initial == ["QUEUED"], f"submit event must refresh cards, got {initial}"

    tree = queue_view.build()
    results = " ".join(text.value or "" for text in find_controls(tree, ft.Text))
    assert "a.wav" in results, f"per-file results missing from {results!r}"
    assert find_button(tree, "Cancel", job.id).disabled is False, "cancel must be on while QUEUED"
    assert find_button(tree, "Pause", job.id).disabled is False, "pause must be on while QUEUED"
    assert find_button(tree, "Resume", job.id).disabled is True, "resume off unless PAUSED"
    assert find_button(tree, "Retry", job.id).disabled is True, "retry off unless FAILED"

    # RUNNING keeps pause/cancel enabled (per-state enablement).
    job.state = JobState.RUNNING
    queue_view.bus.publish(
        AppEvent("job_state", {"job_id": job.id, "state": "running", "error": None})
    )
    tree = queue_view.build()
    assert find_button(tree, "Cancel", job.id).disabled is False, "cancel on while RUNNING"
    assert find_button(tree, "Pause", job.id).disabled is False, "pause on while RUNNING"

    # Terminal states disable cancel/retry.
    job.state = JobState.CANCELLED
    queue_view.bus.publish(
        AppEvent("job_state", {"job_id": job.id, "state": "cancelled", "error": None})
    )
    tree = queue_view.build()
    assert find_button(tree, "Cancel", job.id).disabled is True, "cancel off when CANCELLED"
    assert find_button(tree, "Retry", job.id).disabled is True, "retry off when CANCELLED"


def test_pause_resume_cancel_buttons_call_queue(fake_queue_view):
    view = fake_queue_view

    def click(label):
        button = find_button(view.build(), label, "job-1")
        button.on_click(ft.Event("click", button))

    for label in ("Pause", "Resume", "Cancel"):
        click(label)
    assert view.queue.calls == [
        ("pause", "job-1"),
        ("resume", "job-1"),
        ("cancel", "job-1"),
    ], f"recorded calls: {view.queue.calls}"

    # Up/Down reorder buttons delegate to the queue seam as well.
    def click_icon(icon):
        buttons = [
            button
            for button in find_controls(view.build(), ft.IconButton)
            if button.icon == icon and button.data == "job-1"
        ]
        assert buttons, f"icon button {icon} not found"
        buttons[0].on_click(ft.Event("click", buttons[0]))

    click_icon(ft.Icons.KEYBOARD_ARROW_UP)
    click_icon(ft.Icons.KEYBOARD_ARROW_DOWN)
    assert view.queue.calls[-2:] == [
        ("move_up", "job-1"),
        ("move_down", "job-1"),
    ], f"recorded calls: {view.queue.calls}"


def test_retry_requeues_failed_job(queue_view, fake_queue, failed_job):
    assert failed_job.state == JobState.FAILED, f"fixture state: {failed_job.state}"
    button = find_button(queue_view.build(), "Retry", failed_job.id)
    assert button.disabled is False, "retry must be enabled for FAILED jobs"
    button.on_click(ft.Event("click", button))
    assert failed_job.state == JobState.QUEUED, f"after retry: {failed_job.state}"
    assert failed_job.error is None, f"retry must clear the error: {failed_job.error!r}"


def test_console_filters_by_level(console):
    console.append(AppEvent("log", {"level": "INFO", "text": "info-line-marker"}))
    console.append(AppEvent("log", {"level": "DEBUG", "text": "debug-line-marker"}))
    # Default filter is INFO: the DEBUG line is hidden immediately.
    assert "info-line-marker" in console.output.value, "INFO line must be visible"
    assert "debug-line-marker" not in console.output.value, "DEBUG hidden under INFO filter"

    # Lowering the filter to DEBUG reveals the hidden line again.
    console.level_dropdown.value = "DEBUG"
    console.level_dropdown.on_select(ft.Event("select", console.level_dropdown))
    assert "debug-line-marker" in console.output.value, "DEBUG line visible under DEBUG filter"
    assert console.level_filter == "DEBUG", "level_filter must follow the dropdown"

    # Back to INFO: the DEBUG line is filtered out again.
    console.level_dropdown.value = "INFO"
    console.level_dropdown.on_select(ft.Event("select", console.level_dropdown))
    assert "debug-line-marker" not in console.output.value, "DEBUG hidden again under INFO"
    assert "info-line-marker" in console.output.value, "INFO line still visible"


def test_console_caps_at_2000_lines(console):
    for index in range(2500):
        console.append(AppEvent("log", {"level": "INFO", "text": f"line {index}"}))
    assert len(console.lines) == 2000, f"ring buffer must cap at 2000, got {len(console.lines)}"
    assert "line 500" in console.lines[0][1], f"oldest kept line: {console.lines[0][1]!r}"
    assert "line 0" not in console.output.value, "oldest dropped lines must vanish"
    assert "line 500" in console.output.value, "first retained line must render"
    assert "line 2499" in console.output.value, "newest line must render"


def test_settings_theme_change_persists(settings_view, tmp_path):
    settings_view.theme_dropdown.value = "LIGHT"
    settings_view.save()
    reloaded = ConfigStore(tmp_path / "config.json")
    assert reloaded.get("theme_mode") == "LIGHT", "theme_mode must persist as LIGHT"
    # An invalid stored theme falls back to SYSTEM on construction.
    (tmp_path / "config.json").write_text('{"theme_mode": "NEON"}')
    fresh = SettingsView(ConfigStore(tmp_path / "config.json"), settings_view.bus)
    assert fresh.theme_dropdown.value == "SYSTEM", "unknown theme must clamp to SYSTEM"


def test_runtime_repair_publishes_progress(settings_view, fake_bus, monkeypatch, tmp_path):
    def fake_ensure_runtime(progress):
        progress("venv", 0.1)
        progress("torch", 0.3)
        progress("deps", 0.7)
        return RuntimeInfo(mode="venv", python=tmp_path / "venv" / "bin" / "python")

    monkeypatch.setattr("gui.ui.pages.settings_view.ensure_runtime", fake_ensure_runtime)
    button = settings_view.repair_button
    asyncio.run(button.on_click(ft.Event("click", button)))

    runtime_events = [event for event in fake_bus.events if event.type == "runtime"]
    phases = [event.payload.get("phase") for event in runtime_events]
    assert "venv" in phases, f"venv phase missing: {phases}"
    assert "torch" in phases, f"torch phase missing: {phases}"
    assert "deps" in phases, f"deps phase missing: {phases}"
    assert all("progress" in event.payload for event in runtime_events), runtime_events
    assert settings_view.repair_button.disabled is False, "button re-enabled after repair"
