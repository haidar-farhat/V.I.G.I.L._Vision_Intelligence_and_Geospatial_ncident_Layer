"""PyInstaller entry point: a plain script, because a frozen `__main__` cannot use relative imports."""

import sys

from vigil.interfaces.cli import main

if __name__ == "__main__":
    sys.exit(main())
