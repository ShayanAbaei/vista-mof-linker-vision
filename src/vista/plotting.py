"""
Academic plotting standard (spec section 12).

Every publication figure is a vector PDF with embedded TrueType fonts, and the
underlying plotted data is always saved alongside it (CSV / JSON / NPZ) plus a
companion metadata JSON. Colors are colorblind-accessible (Okabe-Ito), diverging
scales are centered at zero, and ``jet`` is never used.

Use :func:`academic_rc` as a context manager and :func:`save_figure` to write the
figure + its data + metadata atomically with deterministic filenames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc_context

# Single- and double-column journal widths (inches).
SINGLE_COL_W = 3.5
DOUBLE_COL_W = 7.2

# Okabe-Ito colorblind-safe qualitative palette (kept for reference / categorical).
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Cohesive professional palette (seaborn "deep" family) used across figures.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
PRIMARY = "#4C72B0"      # primary accent (bars, point estimates)
ACCENT = "#C44E52"       # secondary accent
NEUTRAL = "#4C4C4C"      # text / error bars

# Per-split colors for the label-distribution figure (no rare/dominant coloring).
SPLIT_COLORS = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}

# Colorblind-safe colormaps (never 'jet').
SEQUENTIAL_CMAP = "viridis"   # magnitudes (attention, confusion counts)
DIVERGING_CMAP = "RdBu_r"     # symmetric effects centered at zero (CB-safe)
ATTENTION_CMAP = "magma"      # spatial label->image overlays

# Group -> marker (shape only; colors are NOT used to encode group in figures).
GROUP_MARKER = {"rare": "o", "intermediate": "s", "dominant": "^"}
# Backwards-compat shim (group color encoding removed for publication figures).
GROUP_STYLE = {g: {"color": PRIMARY, "marker": m} for g, m in GROUP_MARKER.items()}

# High-impact-journal style (Matter / Digital Discovery / npj Comput. Mater.):
# clean Arial/Helvetica sans-serif, concise bold titles, embedded TrueType fonts,
# 600 dpi raster embedding for vector PDFs.
ACADEMIC_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.titlepad": 6,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def academic_rc(**overrides):
    """Context manager applying the academic rcParams (with optional overrides)."""
    rc = dict(ACADEMIC_RC)
    rc.update(overrides)
    return rc_context(rc)


def symmetric_robust_limit(values: np.ndarray, q: float = 99.0) -> float:
    """
    Symmetric color limit for a diverging plot: the ``q``-th percentile of the
    absolute (finite) values. Returns a positive float; falls back to 1.0 if no
    finite data.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(np.abs(v), q))
    return lim if lim > 0 else 1.0


def _write_data(data_path_base: Path, data: Dict) -> Dict[str, str]:
    """Write the plotted data next to the figure. Arrays -> NPZ, tables -> CSV,
    scalars/lists -> JSON. Returns a dict of written paths."""
    written = {}
    arrays = {k: np.asarray(v) for k, v in data.items()
              if isinstance(v, (np.ndarray,)) or (isinstance(v, (list, tuple))
              and len(v) and isinstance(v[0], (list, tuple, np.ndarray)))}
    scalars = {k: v for k, v in data.items() if k not in arrays}

    if arrays:
        npz_path = data_path_base.with_suffix(".npz")
        np.savez_compressed(npz_path, **arrays)
        written["npz"] = str(npz_path)
    if scalars:
        json_path = data_path_base.with_name(data_path_base.name + "_data").with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(scalars, f, indent=2, default=_json_default)
        written["json"] = str(json_path)
    return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_figure(
    fig,
    path_base: Union[str, Path],
    data: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    close: bool = True,
) -> Dict[str, str]:
    """
    Save ``fig`` as a vector PDF and write its plotted data + metadata alongside.

    Parameters
    ----------
    fig : matplotlib Figure
    path_base : str | Path
        Path *without* extension; ``.pdf`` (figure), data files and
        ``*_meta.json`` are derived from it.
    data : dict, optional
        The plotted data to persist (arrays -> NPZ, scalars/lists -> JSON).
    metadata : dict, optional
        Free-form configuration metadata stored in ``<base>_meta.json``.
    """
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(pdf_path)

    written = {"pdf": str(pdf_path)}
    if data:
        written.update(_write_data(path_base, data))
    meta = {"figure": str(pdf_path)}
    if metadata:
        meta.update(metadata)
    meta_path = path_base.with_name(path_base.name + "_meta").with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    written["meta"] = str(meta_path)

    if close:
        plt.close(fig)
    return written
