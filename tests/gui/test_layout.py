"""Layout invariant: no expanding control inside a wrapping Row (gray-box guard).

Probe P1 proved an ``expand`` child in a ``wrap=True`` Row fails to render
on Flet 1.0.1 (Flutter Wrap is not a Flex), graying out the whole region
with no Python-side error. This test walks every view tree and fails on
the structural violation instead of waiting for a screenshot.
"""

from __future__ import annotations

import flet as ft

from gui.core.backend import EnhancementBackend
from gui.core.config import ConfigStore
from gui.core.events import EventBus
from gui.core.jobs import JobQueue
from gui.ui.pages import EnhanceView, LogView, QueueView, SettingsView


class FakeBackend(EnhancementBackend):
    def enhance_chunk(self, audio, cfg):
        return audio

    def cancel(self):
        pass

    def shutdown(self):
        pass


def _iter_tree(control):
    yield control
    for child in getattr(control, "controls", None) or []:
        yield from _iter_tree(child)
    content = getattr(control, "content", None)
    if isinstance(content, ft.Control):
        yield from _iter_tree(content)


def _assert_no_expand_in_wrap(tree, view_name):
    violations = []

    def walk(control, in_wrap):
        wrap_here = isinstance(control, ft.Row) and bool(getattr(control, "wrap", False))
        if in_wrap and getattr(control, "expand", None):
            violations.append(f"{type(control).__name__} expand inside wrap Row")
        for child in getattr(control, "controls", None) or []:
            walk(child, in_wrap or wrap_here)
        content = getattr(control, "content", None)
        if isinstance(content, ft.Control):
            walk(content, in_wrap or wrap_here)

    walk(tree, False)
    assert not violations, f"{view_name}: {violations}"


def test_no_expand_inside_wrap_row(tmp_path):
    """Every view tree must keep expanding controls out of wrapping Rows."""
    cfg = ConfigStore(tmp_path / "config.json")
    bus = EventBus()
    queue = JobQueue(FakeBackend(), bus)
    views = {
        "EnhanceView": EnhanceView(cfg, queue, bus),
        "QueueView": QueueView(queue, bus),
        "LogView": LogView(bus, cfg),
        "SettingsView": SettingsView(cfg, bus),
    }
    for name, view in views.items():
        _assert_no_expand_in_wrap(view.build(), name)
