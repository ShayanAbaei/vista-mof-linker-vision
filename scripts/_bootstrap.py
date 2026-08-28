"""Bootstrap the historical VISTA modules without changing their import semantics.

The imported source tree predates package-relative imports (e.g. ``from masking
import ...``). During the reproducibility phase we intentionally prepend
``src/vista`` to sys.path rather than refactoring all historical imports before
the sample-level reproduction experiment.
"""
from __future__ import annotations
import sys
from pathlib import Path

def add_vista_legacy_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    legacy = repo / "src" / "vista"
    if not legacy.exists():
        raise RuntimeError(f"VISTA source directory not found: {legacy}")
    s = str(legacy)
    if s not in sys.path:
        sys.path.insert(0, s)
    return legacy
