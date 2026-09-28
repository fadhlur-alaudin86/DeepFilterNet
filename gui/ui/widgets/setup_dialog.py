"""First-run runtime setup dialog (Task 12).

Shown by ``gui.main`` before the main window when the current environment
is not ready and flet itself is importable (spec 7.2). ``open()`` runs
``ensure_runtime`` in a ``page.run_thread`` worker; phase progress hops
back to the controls through ``"runtime"`` bus events, dispatched on the
page loop when mounted and inline in headless tests.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import flet as ft

from gui.core.dependency import ensure_runtime
from gui.core.events import AppEvent, EventBus

# Phase keys reported by ensure_runtime, in install order
# (venv creation, torch wheel, remaining dependencies).
PHASE_VENV = "venv"
PHASE_TORCH = "torch"
PHASE_DEPS = "deps"
PHASES = (PHASE_VENV, PHASE_TORCH, PHASE_DEPS)
PHASE_TITLES = {
    PHASE_VENV: "Creating virtual environment",
    PHASE_TORCH: "Installing PyTorch",
    PHASE_DEPS: "Installing dependencies",
}


class SetupDialog:
    """Interactive runtime bootstrap: phase progress, Retry and Skip.

    ``on_retry``/``on_skip`` notify the owner (``gui.main``); the dialog
    re-invokes ``ensure_runtime`` itself on Retry. ``open(progress_cb)``
    starts the worker: *progress_cb* is handed to ``ensure_runtime`` and
    defaults to ``report_progress``, which publishes ``"runtime"`` events
    on the dialog bus. Failures surface the pip stderr tail (carried in
    the ``RuntimeError`` message) as visible error text.
    """

    def __init__(
        self,
        on_retry: Callable[[], None] | None = None,
        on_skip: Callable[[], None] | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.on_retry = on_retry
        self.on_skip = on_skip
        self.bus = bus or EventBus()

        self.phase_labels = {phase: ft.Text(PHASE_TITLES[phase], size=12) for phase in PHASES}
        self.phase_bars = {phase: ft.ProgressBar(value=0.0, expand=True) for phase in PHASES}
        # The error text carries the RuntimeError message verbatim, which
        # includes the pip stderr tail (gui.core.dependency._pip_install).
        self.error_text = ft.Text("", size=12, color=ft.Colors.ERROR, selectable=True)
        self.skip_button = ft.FilledButton("Skip", on_click=self._on_skip)
        self.retry_button = ft.FilledButton("Retry", on_click=self._on_retry)

        self._progress_cb: Callable[[str, float], None] = self.report_progress
        self.root = self._build_tree()
        self.bus.subscribe(self._on_event)

    # ----------------------------------------------------------------- layout

    def _build_tree(self) -> ft.Control:
        """Compose the dialog body: header, one label+bar row per phase."""
        phase_controls: list[ft.Control] = []
        for phase in PHASES:
            phase_controls.append(self.phase_labels[phase])
            phase_controls.append(self.phase_bars[phase])
        return ft.Column(
            [
                ft.Text("Runtime setup", size=18, weight=ft.FontWeight.W_600),
                ft.Text(
                    "DeepFilterNet installs its Python runtime before enhancing audio.",
                    size=12,
                ),
                *phase_controls,
                self.error_text,
                ft.Row(
                    [self.skip_button, self.retry_button],
                    alignment=ft.MainAxisAlignment.END,
                    spacing=12,
                ),
            ],
            spacing=8,
        )

    def build(self) -> ft.Control:
        """Return the dialog body control (mounted by gui.main's ft.run)."""
        return self.root

    # --------------------------------------------------------------- lifecycle

    def open(self, progress_cb: Callable[[str, float], None] | None = None) -> None:
        """Start ``ensure_runtime`` (Retry reuses the stored callback).

        On a mounted dialog the work runs on ``page.run_thread`` with the
        bus bound to the page loop; headless (no page) it runs inline so
        callers observe the outcome synchronously.
        """
        if progress_cb is not None:
            self._progress_cb = progress_cb
        page = self._mounted_page()
        if page is not None:
            # Worker-thread publishes must dispatch on the page loop, the
            # same convention DeepFilterApp uses for the main window.
            self.bus.bind_loop(page.loop)
        self._start(page)

    def report_progress(self, phase: str, value: float) -> None:
        """ensure_runtime progress seam: publish a ``"runtime"`` bus event."""
        self.bus.publish(AppEvent("runtime", {"phase": phase, "progress": value}))

    # ----------------------------------------------------------------- worker

    def _start(self, page) -> None:
        """Reset the phase displays and run ``ensure_runtime`` once."""
        for phase in PHASES:
            self.phase_labels[phase].value = PHASE_TITLES[phase]
            self.phase_bars[phase].value = 0.0
        self.error_text.value = ""
        self.retry_button.disabled = True
        self._refresh()
        if page is None:
            self._worker(self._progress_cb)  # headless: no page to schedule on
        else:
            page.run_thread(self._worker, self._progress_cb)

    def _worker(self, progress_cb: Callable[[str, float], None]) -> None:
        """ensure_runtime body; publishes done/error onto the bus.

        The catch covers the documented failure modes only (pip errors are
        wrapped in RuntimeError by dependency._pip_install; venv creation
        raises OSError or subprocess errors); anything else is a bug and
        surfaces loudly instead of being swallowed. Skip stays enabled as
        the user escape hatch either way.
        """
        try:
            info = ensure_runtime(progress_cb)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            self.bus.publish(AppEvent("runtime", {"phase": "error", "error": str(exc)}))
        else:
            self.bus.publish(
                AppEvent("runtime", {"phase": "done", "mode": getattr(info, "mode", "venv")})
            )

    # ------------------------------------------------------------------ events

    def _on_event(self, event: AppEvent) -> None:
        """Update phase labels/bars and the error tail from bus payloads."""
        if event.type != "runtime":
            return
        phase = event.payload.get("phase")
        if phase in self.phase_bars:
            try:
                value = min(max(float(event.payload.get("progress") or 0.0), 0.0), 1.0)
            except (TypeError, ValueError):
                return
            self.phase_bars[phase].value = value
            self.phase_labels[phase].value = f"{PHASE_TITLES[phase]} - {value * 100:.0f}%"
        elif phase == "error":
            self.error_text.value = str(event.payload.get("error") or "")
            self.retry_button.disabled = False
        elif phase == "done":
            self.retry_button.disabled = True
        self._refresh()

    def _on_retry(self, _event=None) -> None:
        """Notify the owner, then re-invoke ensure_runtime from scratch."""
        if self.on_retry is not None:
            self.on_retry()
        self._start(self._mounted_page())

    def _on_skip(self, _event=None) -> None:
        """Notify the owner; gui.main closes the dialog and continues degraded."""
        if self.on_skip is not None:
            self.on_skip()

    # ---------------------------------------------------------------- helpers

    def _mounted_page(self):
        """Return the page hosting the dialog, or None when headless."""
        try:
            return self.root.page
        except RuntimeError:
            return None

    def _refresh(self) -> None:
        """Push mutations to the client when the dialog is mounted."""
        page = self._mounted_page()
        if page is not None:
            page.update()
