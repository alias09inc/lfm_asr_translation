from __future__ import annotations

import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lfm_app.cli import main as app_main

    app_main()


if __name__ == "__main__":
    main()
