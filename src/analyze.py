"""Single-photon X-ray event extraction.

Port of MATLAB ``sifAnalyze.m`` (analysis path only -- no plotting, no optional
``bcg_new`` re-estimation outputs).  For each frame we subtract a scaled
background, reject cosmics, then isolate genuine single-photon "grains" via a
3x3 neighbourhood test plus connected-component intensity gating, and accumulate
the cleaned signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from collections.abc import Iterable

import numpy as np
from scipy.ndimage import label, sum_labels
from scipy.signal import convolve2d

from .common import common_mode, adu_histogram, NBINS
from .curvature import CurvatureCorrection
from .sif_io import SifFile


def format_meta_header(meta: dict, extra: dict | None = None) -> str:
    """Render a provenance dict into a multi-line text header for ``np.savetxt``.

    Keys are printed one per line as ``key: value``; list values (e.g. source
    files) are expanded across lines. ``extra`` is merged in last (used for the
    column legend).
    """
    items = dict(meta or {})
    if extra:
        items.update(extra)
    lines = ["Tender_Analysis export"]
    for key, val in items.items():
        if isinstance(val, (list, tuple)):
            # Cap long lists (e.g. an 80-energy RIXS source list) to stay readable.
            if len(val) > 6:
                shown = list(val[:3]) + [f"... ({len(val) - 4} more) ..."] + [val[-1]]
            else:
                shown = list(val)
            lines.append(f"{key}: ({len(val)})")
            lines.extend(f"  {v}" for v in shown)
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _hist_positive(values: np.ndarray) -> np.ndarray:
    """Histogram positive values into ``NBINS`` bins by ceil (MATLAB accumarray)."""
    v = values[values > 0]
    if v.size == 0:
        return np.zeros(NBINS, dtype=np.int64)
    idx = np.clip(np.ceil(v).astype(np.int64), 0, NBINS - 1)
    return np.bincount(idx, minlength=NBINS)

# 4-connectivity, matching MATLAB ``bwconncomp(...,4)``.
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
_BOX3 = np.ones((3, 3))


@dataclass
class XESResult:
    """Outputs of :func:`extract_signal` and :meth:`OnePot.run`.

    Attributes
    ----------
    signal:
        Cleaned X-ray signal, summed over frames.  ``(height, width)`` normally,
        or ``(n_files, height, width)`` in scan mode (one plane per file).
    corr_signal:
        ``signal`` after curvature correction (set by :meth:`OnePot.run`; ``None``
        straight out of :func:`extract_signal`).
    raw:
        Summed raw frames, ``(height, width)``.
    total_counts_raw / total_counts_signal / total_counts_common_mode:
        Per-frame spectra, ``(width, n_frames)`` -- column ``k`` is the length-
        ``width`` spectrum of frame ``k`` (raw / cleaned / common-mode-adjusted).
    bcg:
        Background image actually used.
    t:
        Curvature polynomial coefficients (filled in by :meth:`OnePot.run`).
    histograms:
        When ``histograms=True`` was passed to :func:`extract_signal`, a dict of
        ADU-count histograms (each length ``2**16``, index = ADU value) used to
        diagnose and set thresholds.  Keys: ``"xray"`` (extracted grain
        intensities), ``"bkg_free"`` (raw - background), ``"binned"`` (3x3
        neighbourhood sum), ``"raw"`` (raw frames), ``"background"`` (scaled
        background).  Mirrors columns 1-5 of MATLAB ``sifAnalyze``'s ``his``.
    """

    signal: np.ndarray
    raw: np.ndarray
    total_counts_raw: np.ndarray
    total_counts_signal: np.ndarray
    total_counts_common_mode: np.ndarray
    bcg: np.ndarray
    corr_signal: np.ndarray | None = None
    t: np.ndarray | None = None
    scan: bool = False
    scan_data: np.ndarray | None = None  # (width, n_files), filled in scan mode
    histograms: dict[str, np.ndarray] | None = None  # ADU histograms (if requested)
    meta: dict = field(default_factory=dict)  # provenance for exports (set by OnePot)

    def spectrum(self) -> np.ndarray:
        """Length-``width`` summed spectrum of the (corrected) signal."""
        img = self.corr_signal if self.corr_signal is not None else self.signal
        if img.ndim == 3:
            img = img.sum(axis=0)
        return img.sum(axis=0)

    def save_txt(self, path: str) -> str:
        """Write the emission spectrum to ``path`` as commented-header text.

        Two columns: ``pixel`` and ``counts``. The header records provenance from
        :attr:`meta` (source files, thresholds, background mode, ...). Returns the
        path written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        spec = self.spectrum()
        cols = np.column_stack([np.arange(spec.size), spec])
        header = format_meta_header(self.meta, extra={"columns": "pixel counts"})
        np.savetxt(path, cols, header=header, fmt=["%d", "%.8g"])
        return path


def _background_common_mode(bcg: np.ndarray, bcg_adjust: bool) -> float:
    """Zero-peak position of the background image (0 disables scaling)."""
    if not bcg_adjust or np.all(bcg == 0):
        return 0.0
    return common_mode(bcg, refine=True)


def extract_signal(
    files: Iterable[SifFile],
    threshold,
    bcg=0,
    scan_nbrs=None,
    scan: bool = False,
    bcg_adjust: bool = True,
    curvature: CurvatureCorrection | None = None,
    histograms: bool = False,
    verbose: bool = False,
) -> XESResult:
    """Extract summed single-photon X-ray signal from a set of SIF files.

    Parameters
    ----------
    files:
        Iterable of :class:`~onepot.sif_io.SifFile`.
    threshold:
        1-3 element sequence ``[low, xray, hi]`` in ADU.  ``low`` gates the 3x3
        neighbourhood test, ``hi`` (the last element) removes cosmics, and
        ``xray`` (when 3 elements are given) is the minimum per-grain intensity.
    bcg:
        Background image (``(height, width)``) or ``0`` for none.
    scan_nbrs:
        Optional 0-based frame indices to include (``None`` = all).
    scan:
        When true, keep one signal plane per file rather than one grand sum.
    bcg_adjust:
        Scale the background per frame by the common-mode ratio (default true).
    curvature:
        Correction used only to build ``total_counts`` spectra; defaults to
        identity (``t=1``), matching MATLAB where ``sifAnalyze`` is called without
        a fitted ``t``.
    histograms:
        Accumulate diagnostic ADU histograms into ``XESResult.histograms`` (see
        that field's docstring).  Off by default -- it adds a few `bincount`s per
        frame.
    """
    files = list(files)
    threshold = np.atleast_1d(np.asarray(threshold, dtype=float))
    bcg = np.asarray(bcg, dtype=float) if not np.isscalar(bcg) else np.float64(bcg)
    bcg_cm = _background_common_mode(np.asarray(bcg), bcg_adjust)

    if curvature is None:
        curvature = CurvatureCorrection(t=1)  # identity

    total_frames = sum(f.num_frames for f in files)
    selected = _selected_frames(scan_nbrs, total_frames)

    shape = files[0].shape
    width = shape[1]

    raw = np.zeros(shape)
    if scan:
        signal = np.zeros((len(files),) + shape)
    else:
        signal = np.zeros(shape)

    tc_raw = np.zeros((width, total_frames))
    tc_signal = np.zeros((width, total_frames))
    tc_cm = np.zeros((width, total_frames))

    # Diagnostic ADU histograms (mirror sifAnalyze's `his` columns 1-5).
    hist = None
    if histograms:
        hist = {k: np.zeros(NBINS, dtype=np.int64)
                for k in ("xray", "bkg_free", "binned", "raw", "background")}

    counter = -1  # 0-based global frame index
    for file_idx, sif in enumerate(files):
        if verbose:
            start = counter + 1
            end = start + sif.num_frames - 1
            print(f"    ({file_idx + 1}/{len(files)}) {os.path.basename(sif.path)} "
                  f"— frames {start}–{end}")
        for i in range(sif.num_frames):
            counter += 1
            i_scan = counter  # 0-based column into total_counts
            if counter not in selected:
                continue

            image = sif.frame(i)
            raw += image

            cm = common_mode(image, refine=True)
            bcg_adj = (cm / bcg_cm) if bcg_cm > 0 else 1.0
            frame = image - bcg_adj * bcg

            if hist is not None:
                hist["raw"] += adu_histogram(image)          # col 4: raw data
                hist["bkg_free"] += _hist_positive(frame)     # col 2: raw - bcg
                hist["background"] += adu_histogram(bcg_adj * bcg)  # col 5: background

            # cosmic / high-energy rejection
            cosmics = None
            if threshold.size > 1:
                cosmics = frame > threshold[-1]
                frame = np.where(cosmics, 0.0, frame)

            # 3x3 box sum, then neighbourhood gate + dilation
            frame_binned = convolve2d(frame, _BOX3, mode="same")
            if hist is not None:
                hist["binned"] += _hist_positive(frame_binned)  # col 3: 3x3 binned
            low = threshold[0]
            frame_index = (frame > low / 4) & (frame_binned > low)
            frame_index = convolve2d(frame_index.astype(float), _BOX3, mode="same") > 0

            # keep only flagged pixels
            frame = frame * frame_index

            # connected-component intensity gating (needs 3rd threshold)
            if threshold.size > 2 and frame_index.any():
                labels, nlab = label(frame_index, structure=_CONN4)
                if nlab > 0:
                    grain_int = sum_labels(frame, labels, index=np.arange(1, nlab + 1))
                    if hist is not None:
                        # col 1: ALL grain (X-ray event) intensities, clamped to
                        # [1, NBINS] -- MATLAB histograms every grain, then zeros
                        # sub-threshold ones only in `frame`.
                        clamped = np.clip(grain_int, 1, NBINS - 1)
                        hist["xray"] += np.bincount(
                            np.ceil(clamped).astype(np.int64), minlength=NBINS)
                    bad = np.flatnonzero(grain_int < threshold[1]) + 1
                    if bad.size:
                        frame = np.where(np.isin(labels, bad), 0.0, frame)

            if scan:
                signal[file_idx] += frame
            else:
                signal += frame

            # per-frame spectra (curvature-corrected; identity by default)
            tc_raw[:, i_scan] = curvature.apply(image)[0].sum(axis=0)
            tc_cm[:, i_scan] = curvature.apply(image - cm)[0].sum(axis=0)
            tc_signal[:, i_scan] = curvature.apply(frame)[0].sum(axis=0)

    return XESResult(
        signal=signal,
        raw=raw,
        total_counts_raw=tc_raw,
        total_counts_signal=tc_signal,
        total_counts_common_mode=tc_cm,
        bcg=np.asarray(bcg),
        scan=scan,
        histograms=hist,
    )


def _selected_frames(scan_nbrs, total_frames) -> set[int]:
    if scan_nbrs is None or (hasattr(scan_nbrs, "__len__") and len(scan_nbrs) == 0):
        return set(range(total_frames))
    if isinstance(scan_nbrs, int):
        return {scan_nbrs}
    arr = np.asarray(list(scan_nbrs))
    if arr.dtype == bool:
        return set(np.flatnonzero(arr))
    return set(int(x) for x in arr)
