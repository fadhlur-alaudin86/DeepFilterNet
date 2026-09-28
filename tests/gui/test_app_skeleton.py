"""Flet app skeleton tests: views, navigation, selftest and theme (Task 9)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import flet as ft
import pytest

from gui.app import DeepFilterApp
from gui.core.backend import EnhancementBackend
from gui.core.config import ConfigStore
from gui.core.events import EventBus
from gui.core.jobs import JobQueue
from gui.ui.pages import EnhanceView, LogView, QueueView, SettingsView

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeBackend(EnhancementBackend):
    """Backend stand-in: the skeleton never processes audio."""

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


@pytest.fixture
def page():
    fake = FakePage()
    yield fake
    fake.loop.close()


def find_control(control, control_type):
    """Depth-first search for *control_type* inside a nested control tree."""
    if isinstance(control, control_type):
        return control
    for child in getattr(control, "controls", None) or []:
        found = find_control(child, control_type)
        if found is not None:
            return found
    content = getattr(control, "content", None)
    if isinstance(content, ft.Control):
        return find_control(content, control_type)
    return None


def test_all_views_build_controls(tmp_path):
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    queue = JobQueue(FakeBackend(), bus)
    views = [
        EnhanceView(cfg, queue, bus),
        QueueView(queue, bus),
        LogView(bus),
        SettingsView(cfg, bus),
    ]
    for view in views:
        assert isinstance(view.build(), ft.Control)


def test_navigation_has_four_destinations(page, tmp_path):
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    queue = JobQueue(FakeBackend(), bus)
    app = DeepFilterApp(page, cfg, bus, queue)
    root = app.build()
    assert isinstance(root, ft.Control)
    rail = find_control(root, ft.NavigationRail)
    assert rail is not None, "build() must mount a NavigationRail"
    assert [destination.label for destination in rail.destinations] == [
        "Enhance",
        "Queue",
        "Log",
        "Settings",
    ]


def test_selftest_exits_zero():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "DeepFilterNet")}
    proc = subprocess.run(
        [sys.executable, "-m", "gui", "--selftest"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "selftest ok" in proc.stdout


def test_theme_loaded_from_config(page, tmp_path):
    cfg = ConfigStore(tmp_path / "config.json")
    cfg.set("theme_mode", "DARK")
    bus = EventBus()
    queue = JobQueue(FakeBackend(), bus)
    app = DeepFilterApp(page, cfg, bus, queue)
    app.build()
    assert page.theme_mode == ft.ThemeMode.DARK
    assert page.theme is not None
    assert page.theme.color_scheme_seed is not None
