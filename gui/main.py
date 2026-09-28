"""DeepFilterNet GUI entry point.

Bootstrap order (Task 8): parse CLI flags before any flet import,
re-launch into the managed runtime via ``ensure_gui_process()`` (exit when
it relaunched), then either run ``--selftest`` (headless view build) or
start the UI via ``run_ui()``.
"""

from __future__ import annotations

import argparse
import sys


def run_ui() -> None:
    """Start the Flet UI targeting ``gui.app.main``.

    Flet 1.0 renamed the 0.x ``ft.app(target=...)`` entry point to
    ``ft.run(main)``. Deferred imports keep this module stdlib-only at
    load time so a degraded source environment still reaches the
    bootstrap instead of failing on it.
    """
    import flet as ft

    import gui.app

    ft.run(gui.app.main)


def run_selftest() -> None:
    """Build every view against a temporary ConfigStore, then exit 0.

    Verifies imports, view construction and placeholder builds without
    opening a window: ``ft.run`` is never called.
    """
    import tempfile
    from pathlib import Path

    import flet as ft

    from gui.core.backend import InProcessBackend
    from gui.core.config import ConfigStore
    from gui.core.events import EventBus
    from gui.core.jobs import JobQueue
    from gui.ui.pages import EnhanceView, LogView, QueueView, SettingsView

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ConfigStore(Path(tmp) / "config.json")
        bus = EventBus()
        queue = JobQueue(InProcessBackend(), bus)
        views = (
            EnhanceView(cfg, queue, bus),
            QueueView(queue, bus),
            LogView(bus),
            SettingsView(cfg, bus),
        )
        for view in views:
            control = view.build()
            if not isinstance(control, ft.Control):
                raise TypeError(f"{type(view).__name__}.build() returned {type(control)!r}")
    print("selftest ok")
    sys.exit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui", description="DeepFilterNet desktop GUI")
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help="internal: set when relaunched inside the managed runtime venv",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="build all views against a temp config and exit (no window)",
    )
    # Both flags are parsed before any flet import so the child spawned by
    # ensure_gui_process() starts cleanly (--runtime-child is branched on in
    # Task 12; --selftest branches below, after the bootstrap).
    args = parser.parse_args(argv)

    # Deferred import: keep module import stdlib-only so a degraded source
    # environment reaches the bootstrap instead of failing on it.
    from gui.core.dependency import ensure_gui_process

    if ensure_gui_process():
        return 0  # relaunched in the managed runtime; this process must exit
    if args.selftest:
        run_selftest()  # exits the process with status 0
    run_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
