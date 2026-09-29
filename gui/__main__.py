"""Allow ``python -m gui`` to run the GUI entry point."""

import sys

from gui.main import main

if __name__ == "__main__":
    sys.exit(main())
