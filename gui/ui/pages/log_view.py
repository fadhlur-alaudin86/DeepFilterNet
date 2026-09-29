"""Log view: filterable enhancement log console (Task 11).

Wraps the shared ``Console`` widget: bus ``log`` events are forwarded to
the console's ring buffer, and the selected filter level is persisted to
the ``ConfigStore`` under ``log_view_level`` (a display-only filter; the
run/file level ``log_level`` belongs to the Enhance view and the backend).
Pass the shared app ``ConfigStore`` so the filter survives restarts next
to the other persisted keys; without one the default user config path is
used (same file ``main()`` loads).
"""

from __future__ import annotations

import flet as ft

from gui.core.config import ConfigStore
from gui.core.events import AppEvent, EventBus
from gui.ui.widgets.console import Console


class LogView:
    """Log page: filterable enhancement log console.

    The console level control persists to the ConfigStore on every change;
    the view subscribes to the event bus once, at construction.
    """

    def __init__(self, bus: EventBus, cfg: ConfigStore | None = None) -> None:
        self.bus = bus
        self.cfg = cfg if cfg is not None else ConfigStore()
        stored = str(self.cfg.get("log_view_level") or "INFO").upper()
        self.console = Console(
            level_filter=stored,
            on_level_change=self._persist_level,
        )
        bus.subscribe(self._on_event)

    def build(self) -> ft.Control:
        """Compose the Log page: title over the shared console widget."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Log", size=20, weight=ft.FontWeight.W_600),
                    self.console.build(),
                ],
                spacing=8,
                expand=True,
            ),
            expand=True,
        )

    # ------------------------------------------------------------------ events

    def _on_event(self, event: AppEvent) -> None:
        """Feed bus log records into the console ring buffer."""
        if event.type == "log":
            self.console.append(event)

    def _persist_level(self, level: str) -> None:
        """Write the chosen console filter level back to the ConfigStore."""
        self.cfg.set("log_view_level", level)
        self.cfg.save()
