"""Shared low-level helpers used across analysis stages."""

from __future__ import annotations

import numpy as np

# Andor data is 16-bit; histograms span the full ADU range.
NBINS = 2 ** 16


def adu_histogram(frame: np.ndarray) -> np.ndarray:
    """Histogram integer ADU counts of a frame into ``NBINS`` bins.

    Equivalent to MATLAB ``accumarray(round(frame(:)), 1, [2^16, 1])``.
    """
    vals = np.clip(np.round(frame).astype(np.int64).ravel(), 0, NBINS - 1)
    return np.bincount(vals, minlength=NBINS)


def common_mode(frame: np.ndarray, refine: bool = True) -> float:
    """Estimate the per-frame zero-peak position (the "common mode").

    The zero peak is the most frequent ADU value.  When ``refine`` is set, the
    integer argmax is sharpened by fitting a parabola over the +/-3 neighbouring
    bins and taking its vertex -- this is the ``polyfit((-3:3), ...)`` step in
    ``sifAnalyze.m`` / ``sifBatchBackground.m``.  ``sifBatchBackground`` used the
    bare argmax (``refine=False``); ``sifAnalyze`` used the refined value.
    """
    hist = adu_histogram(frame)
    peak = int(np.argmax(hist))
    if not refine:
        return float(peak)

    offsets = np.arange(-3, 4)
    lo, hi = peak - 3, peak + 4
    if lo < 0 or hi > NBINS:
        return float(peak)
    p = np.polyfit(offsets, hist[lo:hi], 2)
    if p[0] == 0:
        return float(peak)
    return -p[1] / (2 * p[0]) + peak
