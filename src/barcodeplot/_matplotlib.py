from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib() -> None:
    if os.access(Path.home(), os.W_OK):
        return

    cache_root = Path(tempfile.gettempdir()) / "barcodeplot-cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
