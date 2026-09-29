"""Ordered input-file list widget for the Enhance view."""

from __future__ import annotations

from pathlib import Path

import flet as ft


class FileList:
    """Selected input files: add paths, render them, expose them as ``Path``s.

    The widget owns its control tree (``self.control``); the owning view
    mounts it with ``build()``. Rows are rebuilt from the stored paths so
    state survives view rebuilds.
    """

    def __init__(self) -> None:
        self._paths: list[Path] = []
        self._count = ft.Text("0 files", size=12)
        self._rows = ft.Column([self._empty_row()], spacing=4, scroll=ft.ScrollMode.AUTO)
        self.clear_button = ft.TextButton("Clear", on_click=self._on_clear)
        self.control = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("Input files", size=14, weight=ft.FontWeight.W_600),
                            self._count,
                            self.clear_button,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self._rows,
                ],
                spacing=8,
                expand=True,
            ),
            border=ft.Border.all(1, "grey400"),
            border_radius=8,
            padding=ft.Padding.all(10),
            expand=True,
        )

    # ------------------------------------------------------------------ state

    def add_paths(self, paths: list[str]) -> None:
        """Append *paths*, skipping duplicates, and re-render the rows."""
        for raw in paths:
            path = Path(raw)
            if path not in self._paths:
                self._paths.append(path)
        self._sync()

    def paths(self) -> list[Path]:
        """Return the selected files in insertion order."""
        return list(self._paths)

    def clear(self) -> None:
        """Drop every selected file."""
        self._paths.clear()
        self._sync()

    def build(self) -> ft.Control:
        """Return the widget's control tree for mounting by the owner view."""
        return self.control

    # --------------------------------------------------------------- internals

    def _empty_row(self) -> ft.Text:
        return ft.Text("No files selected - use Add Files to pick audio.", size=13)

    def _sync(self) -> None:
        """Rebuild rows and the file counter from the stored paths."""
        rows = [ft.Text(str(path), size=13) for path in self._paths]
        self._rows.controls = rows or [self._empty_row()]
        self._count.value = f"{len(self._paths)} file(s)"

    def _on_clear(self, e: ft.Event) -> None:
        self.clear()
