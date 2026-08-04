"""High-level analysis pipelines.

- :class:`OnePot` ports the analysis body of ``onepot.m``: resolve files,
  build/accept a background, extract the X-ray signal, and curvature-correct it.
- :class:`OnePotRIXS` ports the analysis body of ``onepotRIXS.m``: run the scan,
  assemble the RIXS map, and extract a HERFD/XAS line-out.

All plotting, ``.mat`` caching, and interactive (``ginput``) behaviour from the
MATLAB originals is intentionally dropped.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dataclasses_field

import numpy as np
from scipy.optimize import curve_fit

from .analyze import XESResult, extract_signal, format_meta_header
from .background import compute_background
from .curvature import CurvatureCorrection
from .files import find_sif_files
from .sif_io import SifFile


@dataclass
class Thresholds:
    """ADU thresholds controlling event extraction.

    ``[bcg_cutoff, low, xray, hi]`` matching the 4-element MATLAB ``Treshold``.
    """

    bcg_cutoff: float
    low: float
    xray: float
    hi: float

    @classmethod
    def from_input(cls, value) -> "Thresholds":
        """Expand a user threshold (``None`` / scalar / 2- / 3- / 4-element).

        Mirrors the defaulting logic in ``onepot.m`` lines 176-190.
        """
        if value is None or (hasattr(value, "__len__") and len(value) == 0):
            return cls(60, 100, 170, 2000)
        v = np.atleast_1d(np.asarray(value, dtype=float))
        if v.size == 1:
            return cls(v[0] * 0.8, v[0], v[0] * 1.1, 2 ** 16)
        if v.size == 2:
            # MATLAB set positions 2 and 4 from the pair, then filled 1 and 3.
            return cls(v[1] * 0.8, v[1], v[1] * 1.1, v[1])
        if v.size == 3:
            return cls(v[0], v[1], v[1] * 1.1, v[2])
        return cls(v[0], v[1], v[2], v[3])

    def as_array(self) -> np.ndarray:
        return np.array([self.bcg_cutoff, self.low, self.xray, self.hi])


@dataclass
class RIXSResult:
    """HERFD/XAS extraction outputs (analysis-only port of ``onepotRIXS.m``)."""

    E: np.ndarray          # incident-energy axis
    HERFD: np.ndarray      # emission-band line-out vs E
    TFY: np.ndarray        # total fluorescence yield vs E
    central_pix: int       # emission-line centre used for the band
    rixs_map: np.ndarray   # (pixel, energy) map the line-out came from
    meta: dict = dataclasses_field(default_factory=dict)  # provenance for exports

    def save_txt(self, path: str, save_map: bool = False) -> list[str]:
        """Write the HERFD/TFY line-outs to ``path`` as commented-header text.

        Columns: ``energy``, ``HERFD``, ``TFY``. With ``save_map=True`` the full
        ``(pixel, energy)`` RIXS map is also written to ``<path stem>_map.txt``.
        Returns the list of paths written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cols = np.column_stack([self.E, self.HERFD, self.TFY])
        header = format_meta_header(
            self.meta, extra={"central_pix": self.central_pix,
                              "columns": "energy_eV HERFD TFY"})
        np.savetxt(path, cols, header=header, fmt="%.8g")
        written = [path]
        if save_map:
            stem, ext = os.path.splitext(path)
            map_path = f"{stem}_map{ext or '.txt'}"
            map_header = format_meta_header(
                self.meta,
                extra={"content": "RIXS map (rows=emission pixel, cols=incident energy)",
                       "energies_eV": np.array2string(self.E, precision=2)})
            np.savetxt(map_path, self.rixs_map, header=map_header, fmt="%.8g")
            written.append(map_path)
        return written


class OnePot:
    """Single-file / scan XES analysis pipeline.

    Parameters
    ----------
    files:
        Glob pattern, path, or explicit list passed to :func:`find_sif_files`.
    threshold:
        Passed to :meth:`Thresholds.from_input`.
    bcg:
        ``None`` -> compute from the data; ``0`` -> no background; or a
        ``(height, width)`` array to use directly.
    evolution:
        Two-pass mode: a first pass with only the ``bcg_cutoff`` threshold
        estimates curvature, the second pass uses the remaining thresholds.
    scan:
        Keep one signal plane per file and build per-file spectra.
    bcg_adjust:
        Per-frame common-mode scaling of the background.
    file_nbrs / scan_nbrs:
        Optional 0-based file / frame subset selectors (``None`` = all).
    """

    def __init__(
        self,
        files,
        threshold=None,
        bcg=None,
        evolution: bool = False,
        scan: bool = False,
        bcg_adjust: bool = True,
        file_nbrs=None,
        scan_nbrs=None,
        histograms: bool = False,
    ):
        self.files = files
        self.thresholds = Thresholds.from_input(threshold)
        self.bcg_input = bcg
        self.evolution = evolution
        self.scan = scan
        self.bcg_adjust = bcg_adjust
        self.file_nbrs = file_nbrs
        self.scan_nbrs = scan_nbrs
        self.histograms = histograms

        self._paths: list[str] | None = None
        self._sif: list[SifFile] | None = None

    # -- stages -----------------------------------------------------------

    def find_files(self) -> list[str]:
        """Resolve and cache the sorted list of ``.sif`` paths."""
        if self._paths is None:
            self._paths = find_sif_files(self.files, self.file_nbrs)
            self._sif = [SifFile(p) for p in self._paths]
        return self._paths

    @property
    def sif_files(self) -> list[SifFile]:
        self.find_files()
        return self._sif

    def compute_background(self) -> np.ndarray:
        """Resolve the background image per the ``bcg`` constructor argument."""
        if self.bcg_input is None:
            return compute_background(self.sif_files, self.scan_nbrs)
        if np.isscalar(self.bcg_input):
            if self.bcg_input == 0:
                return np.zeros(self.sif_files[0].shape)
            raise ValueError(f"Unsupported scalar bcg: {self.bcg_input!r}")
        return np.asarray(self.bcg_input, dtype=float)

    def analyze(self, bcg, threshold, curvature=None, histograms=False) -> XESResult:
        """Run :func:`extract_signal` over the resolved files."""
        return extract_signal(
            self.sif_files,
            threshold,
            bcg=bcg,
            scan_nbrs=self.scan_nbrs,
            scan=self.scan,
            bcg_adjust=self.bcg_adjust,
            curvature=curvature,
            histograms=histograms,
        )

    def correct(self, image, t=None) -> tuple[np.ndarray, np.ndarray]:
        """Fit (if needed) and apply curvature correction to ``image``."""
        cc = CurvatureCorrection(t=t)
        corrected, _ = cc.fit_apply(image)
        return corrected, cc.t

    # -- orchestration ----------------------------------------------------

    def run(self) -> XESResult:
        """Execute the full pipeline and return an :class:`XESResult`."""
        self.find_files()
        bcg = self.compute_background()
        th = self.thresholds

        if self.evolution:
            # Pass 1: cutoff-only threshold, estimate curvature from the sum.
            first = self.analyze(bcg, [th.bcg_cutoff])
            _, t = self.correct(_sum_planes(first.signal))
            # Pass 2: real thresholds, corrected with the fitted curvature.
            result = self.analyze(bcg, [th.low, th.xray, th.hi],
                                   curvature=CurvatureCorrection(t=t),
                                   histograms=self.histograms)
        else:
            result = self.analyze(bcg, [th.low, th.xray, th.hi],
                                   histograms=self.histograms)
            t = None

        corr_signal, t = self.correct(_sum_planes(result.signal), t=t)
        result.corr_signal = corr_signal
        result.t = t

        if self.scan:
            # Per-file corrected spectra -> (pixel, file) scan map.
            cc = CurvatureCorrection(t=t)
            scan_cols = [cc.apply(result.signal[i])[0].sum(axis=0)
                         for i in range(result.signal.shape[0])]
            result.scan_data = np.array(scan_cols).T  # (width, n_files)

        result.meta = self._provenance()
        return result

    def _provenance(self) -> dict:
        """Provenance dict recorded on the result for text exports."""
        return {
            "pipeline": type(self).__name__,
            "n_files": len(self._paths or []),
            "source_files": [os.path.basename(p) for p in (self._paths or [])],
            "thresholds [bcg_cutoff, low, xray, hi]":
                np.array2string(self.thresholds.as_array(), precision=1),
            "background": self._bcg_description(),
            "bcg_adjust": self.bcg_adjust,
            "evolution": self.evolution,
        }

    def _bcg_description(self) -> str:
        if self.bcg_input is None:
            return "computed from data (min-projection)"
        if np.isscalar(self.bcg_input):
            return "none" if self.bcg_input == 0 else f"scalar {self.bcg_input}"
        return "supplied array"

    # -- energy axis helpers (shared with RIXS) ---------------------------

    def energy_axis(self) -> np.ndarray:
        """Incident energy per file from the ``mono`` comment or the filename."""
        energies = []
        for sif in self.sif_files:
            e = sif.mono
            if np.isnan(e):
                e = _energy_from_name(sif.path)
            energies.append(e)
        return np.asarray(energies, dtype=float)

    def i0_values(self) -> np.ndarray:
        """Per-file I0 monitor values (``nan`` where unavailable)."""
        return np.asarray([s.I0 for s in self.sif_files], dtype=float)


class OnePotRIXS(OnePot):
    """RIXS/HERFD analysis: builds a scan map, then extracts an emission band.

    Always runs in ``scan`` mode.  Use :meth:`herfd` to obtain the line-out.

    Parameters
    ----------
    exclude_dark:
        Drop dark/reference frames from the scan (default ``True``).  The
        acquisition scripts (``controls/rixs.m``, ``controls/acq.m``) name a dark
        by appending ``_dark`` to the file root -- it is taken with the shutter
        closed and is meant to be subtracted as background, not treated as a scan
        point.  Since the shutter state is not recorded in the SIF metadata (and a
        dark can carry as much signal as a real point and even share a real
        point's ``mono`` energy), the ``_dark.sif`` filename suffix is the only
        reliable marker.  Set ``False`` to keep every matched file.
    dark_suffix:
        Case-insensitive filename stem suffix marking a dark file (default
        ``"_dark"``, i.e. ``*_dark.sif``).
    use_dark_as_background:
        Subtract the frame-averaged ``*_dark.sif`` from each scan frame instead of
        computing a min-projection background from the scan itself (default
        ``False``).  The MATLAB ``onepotRIXS.m`` always used the min-projection --
        it never wired up the dark -- but ``onepot.m`` supports a dark file as the
        ``bcg`` argument (lines 146-154), so this exposes that capability for the
        RIXS workflow.  Requires exactly one dark file in the set, and cannot be
        combined with an explicit ``bcg`` argument.
    """

    def __init__(self, files, threshold=None, bcg=None, bcg_adjust=True,
                 file_nbrs=None, scan_nbrs=None, exclude_dark=True, dark_suffix="_dark",
                 use_dark_as_background=False, evolution=False, histograms=False):
        super().__init__(files, threshold=threshold, bcg=bcg, evolution=evolution,
                         scan=True, bcg_adjust=bcg_adjust,
                         file_nbrs=file_nbrs, scan_nbrs=scan_nbrs,
                         histograms=histograms)
        self.exclude_dark = exclude_dark
        self.dark_suffix = dark_suffix
        self.use_dark_as_background = use_dark_as_background
        self._dark_paths: list[str] = []
        if use_dark_as_background and bcg is not None:
            raise ValueError(
                "use_dark_as_background=True conflicts with an explicit bcg argument"
            )

    def _is_dark(self, path: str) -> bool:
        """True if ``path`` is a dark frame per the ``*_dark.sif`` convention."""
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.lower().endswith(self.dark_suffix.lower())

    def find_files(self) -> list[str]:
        """Resolve files, separating dark frames from scan points.

        Dark frames are always recorded in ``self._dark_paths`` (so they can be
        used as a background); they are removed from the returned scan-point list
        unless ``exclude_dark`` is off.
        """
        if self._paths is None:
            paths = find_sif_files(self.files, self.file_nbrs)
            self._dark_paths = [p for p in paths if self._is_dark(p)]
            if self.exclude_dark:
                paths = [p for p in paths if not self._is_dark(p)]
                if self._dark_paths:
                    print(f"OnePotRIXS: excluded {len(self._dark_paths)} dark "
                          f"frame(s) (*{self.dark_suffix}.sif) from the scan")
            self._paths = paths
            self._sif = [SifFile(p) for p in paths]
        return self._paths

    def compute_background(self) -> np.ndarray:
        """Background image, optionally the frame-averaged dark (else base logic)."""
        if not self.use_dark_as_background:
            return super().compute_background()

        self.find_files()
        if len(self._dark_paths) != 1:
            raise ValueError(
                f"use_dark_as_background=True requires exactly one *{self.dark_suffix}.sif "
                f"file; found {len(self._dark_paths)}"
            )
        dark = SifFile(self._dark_paths[0])
        # Average the dark's frames, mirroring onepot.m:146-154.
        bcg = dark.data.mean(axis=0)
        print(f"OnePotRIXS: using {os.path.basename(self._dark_paths[0])} "
              f"as background (mean ADU {bcg.mean():.1f})")
        return bcg

    def herfd(self, central_pix=None, n: int = 3, i0_corr: bool = True) -> RIXSResult:
        """Extract a HERFD/XAS line-out from the RIXS map.

        Parameters
        ----------
        central_pix:
            Centre emission pixel of the band.  If ``None``, located by a
            gaussian fit to the summed emission peak (``onepotRIXS.m`` 138-156).
        n:
            Band width in pixels (rows summed around ``central_pix``).
        i0_corr:
            Divide each energy column by its I0 before the line-out.
        """
        result = self.run()
        rixs = result.scan_data          # (pixel, energy)
        E = self.energy_axis()

        if i0_corr:
            i0 = self.i0_values()
            i0 = np.where((i0 == 0) | np.isnan(i0), 1.0, i0)
            rixs = rixs / i0[np.newaxis, :]

        if central_pix is None:
            central_pix = _fit_central_pixel(rixs)

        rng = np.arange(n) - n // 2
        band = np.clip(central_pix + rng, 0, rixs.shape[0] - 1)
        HERFD = rixs[band, :].sum(axis=0)
        TFY = rixs.sum(axis=0)

        order = np.argsort(E)
        meta = dict(result.meta)
        meta.update({
            "herfd_band_width_px": n,
            "central_pix": int(central_pix),
            "i0_corrected": i0_corr,
            "exclude_dark": self.exclude_dark,
            "use_dark_as_background": self.use_dark_as_background,
        })
        return RIXSResult(
            E=E[order],
            HERFD=HERFD[order],
            TFY=TFY[order],
            central_pix=int(central_pix),
            rixs_map=rixs[:, order],
            meta=meta,
        )


# -- module helpers -------------------------------------------------------

def _sum_planes(signal: np.ndarray) -> np.ndarray:
    """Collapse a scan-mode ``(n_files, h, w)`` stack to a single image."""
    return signal.sum(axis=0) if signal.ndim == 3 else signal


def _energy_from_name(path: str) -> float:
    """Parse a ``dddd.dd`` energy token from a filename (MATLAB fallback)."""
    m = re.search(r"(\d{4}\.\d{2})", path)
    return float(m.group(1)) if m else float("nan")


def _gaussian(x, a, b, c, d):
    return a * np.exp(-((x - b) / c) ** 2) + d


def _fit_central_pixel(rixs: np.ndarray) -> int:
    """Locate the emission-line centre pixel via a gaussian fit.

    Sums the last few energy columns (the elastic/emission region) into a single
    profile and fits a gaussian, as ``onepotRIXS.m`` does before choosing the
    HERFD band.
    """
    ncols = rixs.shape[1]
    band = rixs[:, max(0, ncols - 10):].sum(axis=1)
    x = np.arange(band.size)
    peak = int(np.argmax(band))
    a0 = band[peak] - np.median(band)
    p0 = [a0 if a0 > 0 else band[peak], float(peak), 3.0, float(np.median(band))]
    try:
        popt, _ = curve_fit(_gaussian, x, band, p0=p0, maxfev=10000)
        centre = int(round(popt[1]))
    except (RuntimeError, ValueError):
        centre = peak
    return int(np.clip(centre, 0, band.size - 1))
