"""Filterable log console widget for the DeepFilterNet GUI (Task 11).

Keeps a bounded ring buffer of ``AppEvent("log", ...)`` records, renders
them through a level filter and offers Clear/Copy/Save actions. The
``LogView`` page wraps one of these and persists the selected level.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import flet as ft

from gui.core.events import AppEvent

# Dropdown choices; the filter shows every line at or above the selection.
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# Severity ranks for filtering; unknown runtime levels fall back to INFO.
_LEVEL_RANK = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
_DEFAULT_RANK = 20

SAVE_FILE_NAME = "enhance-log.txt"


class Console:
    """Scrollable log console with a level dropdown and a 2000-line ring buffer.

    ``append(event)`` stores ``(level, rendered)`` pairs in a ``deque`` with
    ``maxlen=MAX_LINES`` so the oldest lines drop out automatically; the
    visible text is re-rendered through the level filter on every change.
    """

    MAX_LINES = 2000

    def __init__(
        self,
        level_filter: str = "INFO",
        on_level_change: Callable[[str], None] | None = None,
    ) -> None:
        candidate = str(level_filter or "").upper()
        self.level_filter = candidate if candidate in LEVELS else "INFO"
        self.on_level_change = on_level_change
        # maxlen makes the deque a ring buffer: appending past the cap
        # silently drops the oldest entry.
        self.lines: deque[tuple[str, str]] = deque(maxlen=self.MAX_LINES)

        self.level_dropdown = ft.Dropdown(
            label="Level",
            value=self.level_filter,
            options=[ft.DropdownOption(key=level, text=level) for level in LEVELS],
            on_select=self._on_level_select,
        )
        self.clear_button = ft.FilledButton("Clear", on_click=self._on_clear)
        self.copy_button = ft.FilledButton("Copy", on_click=self._on_copy)
        self.save_button = ft.FilledButton("Save", on_click=self._on_save)
        # Flet 1.0 services auto-register with the page at construction and
        # are garbage-collected unless a live owner keeps a reference, so
        # both are pinned as attributes (same pattern as EnhanceView).
        self.clipboard = ft.Clipboard()
        self.save_picker = ft.FilePicker()
        self._services = (self.clipboard, self.save_picker)
        self.output = ft.Text(
            value="",
            size=12,
            font_family="monospace",
            selectable=True,
        )

    # ------------------------------------------------------------------ data

    def append(self, event: AppEvent) -> None:
        """Append a ``"log"`` bus event to the ring buffer and re-render."""
        if event.type != "log":
            return
        payload = event.payload or {}
        level = str(payload.get("level") or "INFO").upper()
        text = str(payload.get("text") or "")
        self.lines.append((level, f"{level}: {text}"))
        self._render()
        self._refresh()

    def visible_text(self) -> str:
        """Rendered lines that pass the current level filter, newline-joined."""
        threshold = _LEVEL_RANK.get(self.level_filter, _DEFAULT_RANK)
        return "\n".join(
            rendered
            for level, rendered in self.lines
            if _LEVEL_RANK.get(level, _DEFAULT_RANK) >= threshold
        )

    # ----------------------------------------------------------------- layout

    def build(self) -> ft.Control:
        """Compose the toolbar (level + actions) over the scrollable output."""
        return ft.Column(
            [
                ft.Row(
                    [
                        self.level_dropdown,
                        self.clear_button,
                        self.copy_button,
                        self.save_button,
                    ],
                    spacing=8,
                    wrap=True,
                ),
                ft.Container(
                    content=ft.Column(
                        [self.output],
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    ),
                    expand=True,
                ),
            ],
            spacing=8,
            expand=True,
        )

    # ---------------------------------------------------------------- updates

    def _render(self) -> None:
        self.output.value = self.visible_text()

    def _refresh(self) -> None:
        """Push mutations to the client when running outside a Flet event."""
        try:
            page = self.output.page
        except RuntimeError:
            return  # controls not mounted yet (headless build)
        page.update()

    # ------------------------------------------------------- control handlers

    def _on_level_select(self, e: ft.Event) -> None:
        value = str(self.level_dropdown.value or "").upper()
        self.level_filter = value if value in LEVELS else "INFO"
        self._render()
        self._refresh()
        if self.on_level_change is not None:
            self.on_level_change(self.level_filter)

    def _on_clear(self, e: ft.Event) -> None:
        self.lines.clear()
        self._render()
        self._refresh()

    async def _on_copy(self, e: ft.Event) -> None:
        # Clipboard is a service control: the await returns when the client
        # acknowledged the write (headless tests never reach this path).
        await self.clipboard.set(self.visible_text())

    async def _on_save(self, e: ft.Event) -> None:
        # save_file() returns the chosen path directly (flet 1.0 return-value
        # pattern); None means the dialog was cancelled.
        path = await self.save_picker.save_file(
            dialog_title="Save log",
            file_name=SAVE_FILE_NAME,
        )
        if path:
            Path(path).write_text(self.visible_text(), encoding="utf-8")
