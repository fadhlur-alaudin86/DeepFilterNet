"""Setup dialog, degraded mode and model-download status tests (Task 12)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import flet as ft
import pytest

from gui.app import DEGRADED_BANNER_TEXT, DeepFilterApp
from gui.core.backend import EnhancementBackend
from gui.core.config import ConfigStore
from gui.core.dependency import RuntimeInfo
from gui.core.events import AppEvent, EventBus
from gui.core.jobs import JobQueue
from gui.ui.widgets.setup_dialog import SetupDialog


class FakeBackend(EnhancementBackend):
    """Backend stand-in: these tests never process audio."""

    def enhance_chunk(self, audio, cfg):
        return audio

    def cancel(self):
        pass

    def shutdown(self):
        pass


class FakeWindow:
    """Duck-typed ft.Window stand-in (real Window lives on a live session)."""

    def __init__(self):
        self.width = None
        self.height = None
        self.on_event = None


class FakePage:
    """Duck-typed ft.Page stand-in: a real Page cannot be built headlessly."""

    def __init__(self):
        self.theme_mode = ft.ThemeMode.SYSTEM
        self.theme = None
        self.window = FakeWindow()
        self.loop = asyncio.new_event_loop()
        self.on_resize = None
        self.updates = 0

    def update(self, *controls):
        self.updates += 1


class RecordingBus(EventBus):
    """EventBus that keeps every published event for assertions."""

    def __init__(self):
        super().__init__()
        self.events: list[AppEvent] = []

    def publish(self, event: AppEvent) -> None:
        self.events.append(event)
        super().publish(event)


@pytest.fixture
def page():
    fake = FakePage()
    yield fake
    fake.loop.close()


@pytest.fixture
def fake_bus():
    return RecordingBus()


@pytest.fixture
def degraded_view(page, tmp_path):
    """App built the way gui.main continues after the user skips setup."""
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    app = DeepFilterApp(
        page,
        cfg,
        bus,
        JobQueue(FakeBackend(), bus),
        runtime=RuntimeInfo(mode="degraded", python=Path(sys.executable)),
    )
    app.build()
    return app


@pytest.fixture
def app_subscriber(page, tmp_path):
    """App whose status bar subscribes to the event bus via the page loop."""
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    app = DeepFilterApp(page, cfg, bus, JobQueue(FakeBackend(), bus))
    app.build()
    return app


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


def pump(loop, rounds=5):
    """Advance the headless page loop so loop-bound bus dispatches run."""
    for _ in range(rounds):
        loop.run_until_complete(asyncio.sleep(0))


def test_setup_dialog_reports_phases(fake_bus):
    # simulate progress("venv", .1), ("torch", .5) -> labels updated
    dialog = SetupDialog(bus=fake_bus)
    dialog.build()

    def fake_ensure_runtime(progress):
        progress("venv", 0.1)
        progress("torch", 0.5)
        return RuntimeInfo(mode="venv", python=Path("venv/bin/python"))

    with patch("gui.ui.widgets.setup_dialog.ensure_runtime", fake_ensure_runtime):
        dialog.open()  # headless: no page, so the worker runs inline

    assert dialog.phase_bars["venv"].value == 0.1, "venv bar must show 10%"
    assert dialog.phase_bars["torch"].value == 0.5, "torch bar must show 50%"
    assert "10%" in dialog.phase_labels["venv"].value, dialog.phase_labels["venv"].value
    assert "50%" in dialog.phase_labels["torch"].value, dialog.phase_labels["torch"].value
    phases = [event.payload.get("phase") for event in fake_bus.events if event.type == "runtime"]
    assert "venv" in phases, f"venv progress must reach the bus: {phases}"
    assert "torch" in phases, f"torch progress must reach the bus: {phases}"


def test_setup_dialog_shows_error_and_retry(tmp_path):
    # ensure_runtime raises RuntimeError("pip failed tail") -> error text
    # visible, retry callback re-invokes ensure_runtime
    calls: list[int] = []

    def failing_ensure_runtime(progress):
        calls.append(len(calls))
        progress("venv", 0.1)
        raise RuntimeError("pip failed tail")

    dialog = SetupDialog()
    dialog.build()
    with patch("gui.ui.widgets.setup_dialog.ensure_runtime", failing_ensure_runtime):
        dialog.open()
        assert "pip failed tail" in dialog.error_text.value, dialog.error_text.value
        assert dialog.retry_button.disabled is False, "Retry must be enabled after failure"
        dialog.retry_button.on_click(ft.Event("click", dialog.retry_button))

    assert len(calls) == 2, f"Retry must re-invoke ensure_runtime, calls={calls}"
    assert "pip failed tail" in dialog.error_text.value, "error stays visible after retry"


def test_skip_disables_enhance(degraded_view):
    # runtime degraded -> Enhance All button disabled + banner visible
    app = degraded_view
    enhance_view = app.views[0]
    assert enhance_view.enhance_button.disabled is True, "degraded runtime must disable Enhance All"
    tree = app.build()
    banner = next(
        (text for text in find_controls(tree, ft.Text) if text.value == DEGRADED_BANNER_TEXT),
        None,
    )
    assert banner is not None, "degraded banner must be rendered"
    assert banner.visible is True, "degraded banner must be visible"


def test_model_download_status_event(app_subscriber):
    # publish runtime event phase=model_download -> status bar text contains "Preparing model"
    app = app_subscriber
    app.bus.publish(AppEvent("runtime", {"phase": "model_download"}))
    pump(app.page.loop)
    assert "Preparing model" in app.status_text.value, app.status_text.value

    # Emission seam: the first queued job (Enhance submit) publishes the event.
    app.status_text.value = "Ready"
    app.bus.publish(AppEvent("job_state", {"job_id": "job-1", "state": "queued", "error": None}))
    pump(app.page.loop)
    assert "Preparing model" in app.status_text.value, app.status_text.value


def test_runtime_ready_lifts_degraded(degraded_view):
    """Repair success (runtime phase=ready) re-enables Enhance, no restart."""
    app = degraded_view
    assert app.views[0].enhance_button.disabled is True
    app.bus.publish(AppEvent("runtime", {"phase": "ready"}))
    pump(app.page.loop)
    assert app.degraded is False
    assert app.views[0].enhance_button.disabled is False
    assert app.degraded_banner.visible is False
    assert app.status_text.value == "Ready"
