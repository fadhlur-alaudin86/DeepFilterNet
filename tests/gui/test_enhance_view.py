"""Enhance view tests: snapshot mapping, validation, submit and progress (Task 10)."""

from __future__ import annotations

import flet as ft
import pytest

from gui.core.backend import JobConfig
from gui.core.config import ConfigStore
from gui.core.events import AppEvent, EventBus
from gui.core.models import ModelError
from gui.ui.pages.enhance_view import EnhanceView


class FakeQueue:
    """JobQueue stand-in: records start()/submit() without a worker thread."""

    def __init__(self):
        self.start_calls = 0
        self.submitted: list[tuple[list, JobConfig]] = []
        self.jobs: list = []

    def start(self):
        self.start_calls += 1

    def submit(self, files, cfg):
        self.submitted.append((list(files), cfg))
        return object()


@pytest.fixture
def fake_queue():
    return FakeQueue()


@pytest.fixture
def view(fake_queue, tmp_path):
    return EnhanceView(ConfigStore(tmp_path / "config.json"), fake_queue, EventBus())


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


def test_snapshot_maps_controls_to_jobconfig(view):
    view.atten_slider.value = 12
    view.model_dropdown.value = "DeepFilterNet2"
    view.device_dropdown.value = "CPU"
    cfg = view.snapshot()
    assert isinstance(cfg, JobConfig)
    assert cfg.atten_lim_db == 12
    assert cfg.model == "DeepFilterNet2"
    assert cfg.device == "CPU"
    # Custom paths are validated at snapshot time (Task 3 contract).
    view.model_dropdown.value = "Custom path"
    view.custom_model_field.value = str(view.cfg.path.parent / "missing-model")
    with pytest.raises(ModelError):
        view.snapshot()


def test_atten_zero_maps_to_none(view):
    view.atten_slider.value = 0
    assert view.snapshot().atten_lim_db is None


def test_chunk_size_validation_error_shown(view):
    view.file_list.add_paths([str(view.cfg.path.parent / "a.wav")])
    view.chunk_mode_dropdown.value = "Custom"
    view.chunk_size_field.value = "3"
    assert view.validate() is False
    assert "between 5 and 600" in (view.chunk_size_field.error or "")
    assert view.enhance_all() is False
    assert view.queue.submitted == []


def test_add_files_updates_list_and_persists(view, tmp_path):
    files = [tmp_path / "a.wav", tmp_path / "b.wav"]
    view.file_list.add_paths([str(path) for path in files])
    assert view.file_list.paths() == files
    view.output_dir_field.value = str(tmp_path / "out")
    view.persist_settings()
    assert ConfigStore(view.cfg.path).get("output_dir") == str(tmp_path / "out")


def test_enhance_all_submits_job(view, fake_queue):
    base = view.cfg.path.parent
    view.file_list.add_paths([str(base / "a.wav"), str(base / "b.wav")])
    assert view.enhance_all() is True
    assert fake_queue.start_calls == 1
    assert len(fake_queue.submitted) == 1
    files, cfg = fake_queue.submitted[0]
    assert len(files) == 2
    assert isinstance(cfg, JobConfig)


def test_device_dropdown_populated(monkeypatch):
    monkeypatch.setattr(
        "gui.ui.pages.enhance_view.available_devices",
        lambda: ["Auto", "CPU", "CUDA:0"],
    )
    view = EnhanceView(ConfigStore(), FakeQueue(), EventBus())
    assert [option.key for option in view.device_dropdown.options] == [
        "Auto",
        "CPU",
        "CUDA:0",
    ]


def test_progress_event_updates_bars(view):
    view.bus.publish(AppEvent("job_progress", {"job_id": "job-1", "progress": 0.5}))
    assert view.global_progress.value == 0.5
    bars = find_controls(view.build(), ft.ProgressBar)
    assert view.file_progress in bars
    assert view.global_progress in bars
    view.bus.publish(AppEvent("job_state", {"job_id": "job-1", "state": "failed", "error": "boom"}))
    assert "boom" in view.status_text.value
    # File pickers must stay strongly referenced: Flet's post-event service
    # GC unregisters services whose refcount shows no live owner.
    assert set(view._pickers) == {view.file_picker, view.dir_picker}
