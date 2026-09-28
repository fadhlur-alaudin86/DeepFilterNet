"""DeepFilterNet GUI entry point.

Bootstrap order (Tasks 8/12): parse CLI flags before any flet import,
re-launch into the managed runtime via ``ensure_gui_process()`` when the
environment cannot even show the setup dialog (flet/platformdirs/loguru
missing), run the interactive ``SetupDialog`` before the main window when
flet is importable but the runtime is incomplete, then either run
``--selftest`` (headless view build) or start the UI via ``run_ui()``.
``--runtime-child`` marks the relaunched child and skips the whole
bootstrap (relaunch-loop guard: the child already runs in the ready
runtime).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui.core.dependency import RuntimeInfo


def run_ui(runtime: RuntimeInfo | None = None) -> None:
    """Start the Flet UI targeting ``gui.app.main``.

    Flet 1.0 renamed the 0.x ``ft.app(target=...)`` entry point to
    ``ft.run(main)``. Deferred imports keep this module stdlib-only at
    load time so a degraded source environment still reaches the
    bootstrap instead of failing on it. *runtime* forwards the
    bootstrap's RuntimeInfo (e.g. degraded after a setup skip); None
    lets the app derive readiness from the environment.
    """
    import flet as ft

    import gui.app

    if runtime is None:
        ft.run(gui.app.main)
    else:
        ft.run(lambda page: gui.app.main(page, runtime=runtime))


def run_selftest() -> None:
    """Build every view against a temporary ConfigStore, then exit 0.

    Verifies imports, view construction and placeholder builds without
    opening a window: ``ft.run`` is never called.
    """
    import tempfile

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


def _setup_dialog_available() -> bool:
    """True when the interactive setup dialog can run in this process.

    The dialog needs flet (window) plus the modules ``setup_dialog``
    imports (platformdirs, loguru); a bare source environment may lack
    any of them, in which case bootstrap falls back to the silent
    relaunch path, which installs everything via ``ensure_gui_process``.
    """
    try:
        importlib.import_module("gui.ui.widgets.setup_dialog")
    except ImportError:
        return False
    return True


def run_setup() -> str:
    """Run the first-run setup dialog in its own minimal Flet app.

    Returns ``"done"`` when ``ensure_runtime`` completed (the caller then
    relaunches into the managed runtime) or ``"skip"`` when the user chose
    Skip or closed the dialog (the caller continues degraded).
    """
    import flet as ft

    from gui.ui.widgets.setup_dialog import SetupDialog

    outcome = {"result": "skip"}

    def page_main(page: ft.Page) -> None:
        def finish(result: str) -> None:
            outcome["result"] = result
            page.window.close()

        def on_skip() -> None:
            finish("skip")

        def on_bus_event(event) -> None:
            if event.type == "runtime" and event.payload.get("phase") == "done":
                finish("done")

        # The dialog re-invokes ensure_runtime itself on Retry, so no
        # owner-side on_retry hook is needed here.
        dialog = SetupDialog(on_retry=None, on_skip=on_skip)
        dialog.bus.subscribe(on_bus_event)
        page.title = "DeepFilterNet setup"
        page.add(dialog.build())
        page.update()
        dialog.open()

    ft.run(page_main)
    return outcome["result"]


def _run_degraded() -> int:
    """Continue into the main window with a degraded runtime (spec 7.2).

    The app disables Enhance and shows a banner until the runtime is
    repaired in Settings.
    """
    from gui.core.dependency import RuntimeInfo

    runtime = RuntimeInfo(mode="degraded", python=Path(sys.executable))
    try:
        run_ui(runtime=runtime)
    except ImportError as exc:
        # The skeleton needs torch at import time (gui.core.backend); in
        # such a bare environment a skip cannot show a window at all.
        print(f"GUI cannot start without the runtime: {exc}", file=sys.stderr)
        print("Run the GUI again to retry the runtime setup.", file=sys.stderr)
        return 1
    return 0


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
    # ensure_gui_process() starts cleanly (--runtime-child branches below as
    # the relaunch-loop guard; --selftest branches after the bootstrap).
    args = parser.parse_args(argv)

    if args.runtime_child:
        # Relaunch-loop guard (Task 12): this child already runs inside the
        # managed runtime, so skip the setup dialog and go straight to the UI.
        if args.selftest:
            run_selftest()  # exits the process with status 0
        run_ui()
        return 0

    # Deferred import: keep module import stdlib-only so a degraded source
    # environment reaches the bootstrap instead of failing on it.
    from gui.core.dependency import current_env_ready, ensure_gui_process

    if not current_env_ready():
        if not args.selftest and _setup_dialog_available():
            # Interactive first-run path (Task 12): flet is importable here,
            # so offer the setup dialog before the main window (spec 7.2).
            if run_setup() == "skip":
                return _run_degraded()
            # Setup finished: ensure_gui_process re-verifies (idempotent) and
            # relaunches the child in the managed runtime; exit when it did.
            if ensure_gui_process():
                return 0
            run_ui()
            return 0
        # Silent path (--selftest, or the dialog cannot run here because
        # flet/platformdirs/loguru are missing): ensure the managed runtime
        # and relaunch silently; a ready environment never reaches this.
        if ensure_gui_process():
            return 0  # relaunched in the managed runtime; this process must exit
    if args.selftest:
        run_selftest()  # exits the process with status 0
    run_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
