"""Queue view: job cards with per-job controls (Task 11).

Renders one card per job in ``JobQueue`` order: state chip, progress,
per-file results and Pause/Resume/Cancel/Retry buttons enabled per state,
plus Up/Down reorder buttons that delegate to ``JobQueue.move_up`` /
``move_down``. The view subscribes to the event bus so ``job_state`` and
``job_progress`` payloads rebuild the cards live.
"""

from __future__ import annotations

from pathlib import Path

import flet as ft

from gui.core.events import AppEvent, EventBus
from gui.core.jobs import Job, JobQueue, JobState

# Terminal states: no further queue action applies to the job.
_TERMINAL = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)

# Shown as a placeholder card while the queue holds no jobs.
EMPTY_QUEUE_HINT = "No jobs yet - add audio files in Enhance and press Enhance All."


class QueueView:
    """Queue page: job cards with per-job controls.

    Controls are created per render in ``_job_card``; ``refresh()`` swaps
    the card list in the stable ``cards`` column. The view subscribes to
    the event bus once, at construction (same pattern as EnhanceView).
    """

    def __init__(self, queue: JobQueue, bus: EventBus) -> None:
        self.queue = queue
        self.bus = bus
        self.cards = ft.Column([], spacing=8, scroll=ft.ScrollMode.AUTO, expand=True)
        bus.subscribe(self._on_event)
        self.refresh()

    # ----------------------------------------------------------------- layout

    def build(self) -> ft.Control:
        """Compose the Queue page (spec 4)."""
        self.refresh()
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Queue", size=20, weight=ft.FontWeight.W_600),
                    self.cards,
                ],
                spacing=8,
                expand=True,
            ),
            expand=True,
        )

    def refresh(self) -> None:
        """Rebuild every job card from the queue's current order and states."""
        jobs = self.queue.jobs
        last_index = len(jobs) - 1
        if jobs:
            self.cards.controls = [
                self._job_card(job, index, last_index) for index, job in enumerate(jobs)
            ]
        else:
            self.cards.controls = [self._empty_card()]
        self._refresh()

    def _empty_card(self) -> ft.Control:
        """Placeholder card shown while the queue holds no jobs."""
        return ft.Container(
            content=ft.Text(EMPTY_QUEUE_HINT, size=13),
            padding=ft.Padding.all(10),
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=ft.BorderRadius.all(8),
        )

    # ------------------------------------------------------------------ cards

    def _job_card(self, job: Job, index: int, last_index: int) -> ft.Control:
        """One bordered card: header (chip, progress, actions) + file results."""
        header = ft.Row(
            [
                ft.Chip(label=ft.Text(job.state.value.upper())),
                ft.Text(f"{len(job.files)} file(s)", size=12),
                ft.ProgressBar(value=job.progress, width=140),
                ft.Text(f"{job.progress * 100:.0f}%", size=12),
                *self._action_buttons(job),
                ft.IconButton(
                    ft.Icons.KEYBOARD_ARROW_UP,
                    tooltip="Move up",
                    data=job.id,
                    disabled=index == 0,
                    on_click=self._on_move_up,
                ),
                ft.IconButton(
                    ft.Icons.KEYBOARD_ARROW_DOWN,
                    tooltip="Move down",
                    data=job.id,
                    disabled=index == last_index,
                    on_click=self._on_move_down,
                ),
            ],
            spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        file_lines = [
            ft.Text(
                f"{Path(path).name}: {job.file_results.get(str(path), 'pending')}",
                size=12,
            )
            for path in job.files
        ]
        error_line = []
        if job.state == JobState.FAILED and job.error:
            error_line = [ft.Text(f"Error: {job.error}", size=12, color=ft.Colors.RED_400)]
        return ft.Container(
            content=ft.Column([header, *file_lines, *error_line], spacing=4),
            padding=ft.Padding.all(10),
            border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=ft.BorderRadius.all(8),
        )

    def _action_buttons(self, job: Job) -> list[ft.Control]:
        """Pause/Resume/Cancel/Retry buttons enabled according to job state."""
        state = job.state
        return [
            ft.FilledButton(
                "Pause",
                data=job.id,
                disabled=state not in (JobState.QUEUED, JobState.RUNNING),
                on_click=self._on_pause,
            ),
            ft.FilledButton(
                "Resume",
                data=job.id,
                disabled=state != JobState.PAUSED,
                on_click=self._on_resume,
            ),
            ft.FilledButton(
                "Cancel",
                data=job.id,
                disabled=state in _TERMINAL,
                on_click=self._on_cancel,
            ),
            ft.FilledButton(
                "Retry",
                data=job.id,
                disabled=state != JobState.FAILED,
                on_click=self._on_retry,
            ),
        ]

    # ------------------------------------------------------------------ events

    def _on_event(self, event: AppEvent) -> None:
        """Rebuild the cards when a job's state or progress changes."""
        if event.type in ("job_state", "job_progress"):
            self.refresh()

    def _refresh(self) -> None:
        """Push mutations to the client when running outside a Flet event.

        Bus callbacks and fixture-driven refreshes are not control events,
        and headless tests have no page.
        """
        try:
            page = self.cards.page
        except RuntimeError:
            return  # controls not mounted yet (headless build)
        page.update()

    # ------------------------------------------------------- control handlers

    def _on_pause(self, e: ft.Event) -> None:
        self.queue.pause(e.control.data)
        self.refresh()

    def _on_resume(self, e: ft.Event) -> None:
        self.queue.resume(e.control.data)
        self.refresh()

    def _on_cancel(self, e: ft.Event) -> None:
        self.queue.cancel(e.control.data)
        self.refresh()

    def _on_retry(self, e: ft.Event) -> None:
        self.queue.retry(e.control.data)
        self.refresh()

    def _on_move_up(self, e: ft.Event) -> None:
        self.queue.move_up(e.control.data)
        self.refresh()

    def _on_move_down(self, e: ft.Event) -> None:
        self.queue.move_down(e.control.data)
        self.refresh()
