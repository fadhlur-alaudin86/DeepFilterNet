"""DeepFilterNet GUI entry point.

Bootstrap order (Task 8): parse ``--runtime-child`` before any flet import,
re-launch into the managed runtime via ``ensure_gui_process()`` (exit when
it relaunched), then start the UI. The UI startup half is Task 9's: it
replaces the ``run_ui()`` stub below and extends the argument parser.
"""

from __future__ import annotations

import argparse
import sys


def run_ui() -> None:
    """Start the Flet UI.

    STUB: Task 9 replaces this body with the Flet startup
    (``ft.app(target=gui.app.main)``). Keep the bootstrap order in
    ``main()`` untouched.
    """
    raise NotImplementedError("run_ui() is implemented in Task 9")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui", description="DeepFilterNet desktop GUI")
    parser.add_argument(
        "--runtime-child",
        action="store_true",
        help="internal: set when relaunched inside the managed runtime venv",
    )
    # Parsed (accepted, not rejected) before any flet import so the child
    # spawned by ensure_gui_process() starts cleanly; no branching on the
    # flag yet (Task 9/12 extend this parser).
    parser.parse_args(argv)

    # Deferred import: keep module import stdlib-only so a degraded source
    # environment reaches the bootstrap instead of failing on it.
    from gui.core.dependency import ensure_gui_process

    if ensure_gui_process():
        return 0  # relaunched in the managed runtime; this process must exit
    run_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
