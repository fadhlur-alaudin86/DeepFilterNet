"""LogView regression tests: bus display and level-key split (final review T11-2)."""

from __future__ import annotations

import flet as ft

from gui.core.config import ConfigStore
from gui.core.events import AppEvent, EventBus
from gui.ui.pages import LogView


def test_published_log_event_reaches_console(tmp_path):
    """A bus "log" event must appear in the console buffer and render."""
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    view = LogView(bus, cfg)
    assert isinstance(view.build(), ft.Control)
    bus.publish(AppEvent("log", {"level": "INFO", "text": "display-marker"}))
    assert any("display-marker" in rendered for _, rendered in view.console.lines)
    assert "display-marker" in view.console.visible_text()


def test_filter_level_key_independent_from_run_level(tmp_path):
    """Console filter persists under log_view_level; run log_level untouched."""
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    view = LogView(bus, cfg)
    view.console.level_dropdown.value = "DEBUG"
    view.console._on_level_select(None)
    assert cfg.get("log_view_level") == "DEBUG"
    assert cfg.get("log_level") == "INFO"
    assert view.console.level_filter == "DEBUG"
