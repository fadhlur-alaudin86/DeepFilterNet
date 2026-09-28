"""Queue view placeholder (job cards land in Task 11)."""

from __future__ import annotations

import flet as ft

from gui.core.events import EventBus
from gui.core.jobs import JobQueue


class QueueView:
    """Queue page: job cards with per-job controls (Task 11)."""

    def __init__(self, queue: JobQueue, bus: EventBus) -> None:
        self.queue = queue
        self.bus = bus

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Queue", size=20, weight=ft.FontWeight.W_600),
                    ft.Text(
                        "Placeholder - job cards arrive in Task 11.",
                        size=13,
                    ),
                ],
                spacing=8,
            ),
            expand=True,
        )
