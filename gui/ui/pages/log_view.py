"""Log view placeholder (console widget lands in Task 11)."""

from __future__ import annotations

import flet as ft

from gui.core.events import EventBus


class LogView:
    """Log page: filterable enhancement log console (Task 11)."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Log", size=20, weight=ft.FontWeight.W_600),
                    ft.Text(
                        "Placeholder - log console arrives in Task 11.",
                        size=13,
                    ),
                ],
                spacing=8,
            ),
            expand=True,
        )
