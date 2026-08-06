"""Pixel -> energy calibration from elastic-scattering scans.

The energy-dispersive (width) axis of the spectrometer is calibrated by
collecting *elastic* scattering at several fixed monochromator energies: at each
mono energy the elastically-scattered line lands on a particular detector pixel,
so fitting each elastic peak's centre pixel and then linear-fitting
``(centre_pixel -> mono_energy)`` yields the ``energy = m*pixel + b`` conversion.

This is deliberately a **standalone, skippable** step: elastic data is often not
available until post-beamtime analysis, so it is never built or required by the
standard :mod:`onepot.pipeline` / :func:`onepot.dataset.index_beamtime` workflow.
Build a calibration here when the elastic data exists, persist it with
:meth:`ElasticCalibration.save_json`, and optionally apply it at export time
(``result.save_txt(path, calibration=cal)``).

The fit math is ported from the reference ``tender_tools.py``
(``Tender.fit_peaks`` / ``fit_energy_pixel`` / ``make_energy_axis``): a
Lorentzian per elastic peak, then a linear pixel->energy fit. Elastic ``.sif``
files are recognised via :attr:`onepot.dataset.FileRecord.is_elastic` (they are a
subset of the aux files that ``index_beamtime`` skips from normal analysis).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
from natsort import natsorted
from scipy.optimize import curve_fit

from .dataset import parse_sif_name
from .pipeline import OnePot, _energy_from_name
from .sif_io import SifFile


def _lorentzian(x, amp, center, width):
    """Lorentzian peak (ported from ``tender_tools.Tender.lorentzian``)."""
    return amp * (width ** 2) / ((x - center) ** 2 + width ** 2)


def _linear(x, m, b):
    """Linear model (ported from ``tender_tools.Tender.linear_model``)."""
    return m * x + b


@dataclass
class ElasticPoint:
    """One elastic-scattering measurement: a mono energy and its spectrum.

    ``spectrum`` is the length-``width`` (typically 2048) emission profile summed
    over all frames of the ``.sif`` file; ``energy`` is the monochromator energy
    (from the SIF ``mono`` comment, falling back to the filename).
    """

    energy: float
    spectrum: np.ndarray
    path: str


def index_elastic(directory: str, *, recursive: bool = False, threshold=None,
                  verbose: bool = False) -> list[ElasticPoint]:
    """Find elastic-scattering ``.sif`` files in ``directory`` and load them.

    Globs ``.sif`` files (recursively if ``recursive``), keeps only those whose
    parsed name is flagged :attr:`~onepot.dataset.FileRecord.is_elastic`, and
    returns one :class:`ElasticPoint` per elastic energy (naturally sorted). The
    energy is taken from the SIF ``mono`` comment, falling back to a ``dddd.dd``
    token in the filename; files whose energy cannot be resolved are skipped.

    Each spectrum is produced by the :class:`~onepot.pipeline.OnePot`
    single-photon extraction (not a raw frame sum -- the raw readout baseline
    swamps the sparse elastic line), with the paired ``*_dark.sif`` at the same
    mono energy wired in as the background.

    A matching dark is **required** for every elastic scan: pairing an elastic
    scan with the dark taken at the same energy is the calibration protocol and
    gives the best S/N and energy accuracy. An elastic scan with no dark at its
    energy raises :class:`ValueError` rather than running without a background.

    Parameters
    ----------
    threshold:
        ADU thresholds forwarded to :class:`~onepot.pipeline.OnePot` (see
        :meth:`~onepot.pipeline.Thresholds.from_input`). ``None`` uses the OnePot
        default.
    verbose:
        Forwarded to :class:`~onepot.pipeline.OnePot` for per-file progress.

    A single glob pattern is also accepted in place of a directory.

    Raises
    ------
    ValueError
        If an elastic scan has no ``*_dark.sif`` at the same mono energy.
    """
    if glob.has_magic(directory):
        paths = glob.glob(directory, recursive=recursive)
    elif recursive:
        paths = glob.glob(os.path.join(directory, "**", "*.sif"), recursive=True)
    else:
        paths = glob.glob(os.path.join(directory, "*.sif"))

    # Separate elastic signal files from their paired darks (both carry
    # "elastic" in the name, so is_elastic alone would wrongly include darks).
    recs = [(p, parse_sif_name(p)) for p in natsorted(paths)]
    recs = [(p, r) for p, r in recs if r is not None and r.is_elastic]
    darks = {}
    for p, r in recs:
        if r.is_dark and r.energy is not None:
            darks[round(r.energy, 2)] = p

    points: list[ElasticPoint] = []
    for path, rec in recs:
        if rec.is_dark:
            continue
        # Prefer the SIF mono comment; fall back to the parsed / filename energy.
        energy = SifFile(path).mono
        if energy is None or np.isnan(energy):
            energy = rec.energy if rec.energy is not None else _energy_from_name(path)
        if energy is None or np.isnan(energy):
            continue
        # Single-photon extraction with the paired dark (same energy) as bcg.
        # A matching dark is mandatory -- refuse to calibrate without it.
        dark_path = darks.get(round(float(energy), 2))
        if dark_path is None:
            raise ValueError(
                f"No *_dark.sif at {energy:g} eV to pair with elastic scan "
                f"{os.path.basename(path)!r}. Elastic calibration requires a dark "
                f"at each energy (best S/N and accuracy); add the missing dark or "
                f"drop that energy from the calibration set."
            )
        bcg = SifFile(dark_path).data.mean(axis=0)
        spectrum = OnePot(path, bcg=bcg, threshold=threshold,
                          verbose=verbose).run().spectrum()
        points.append(ElasticPoint(energy=float(energy),
                                    spectrum=spectrum, path=path))
    return points


@dataclass
class ElasticCalibration:
    """Linear pixel -> energy calibration ``energy = m*pixel + b``.

    Build one with :meth:`fit` (or :func:`calibrate_from_directory`), apply it via
    :meth:`to_energy`, and persist it with :meth:`save_json` / :meth:`load_json`.
    ``centers`` / ``energies`` are the fitted elastic peak pixels and their mono
    energies; ``rms`` is the RMS residual (eV) of the linear fit -- a quick
    goodness check.
    """

    m: float
    b: float
    centers: np.ndarray = field(default_factory=lambda: np.array([]))
    energies: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    rms: float = float("nan")
    paths: list[str] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def fit(cls, points: list[ElasticPoint], *,
            fit_window: int | None = None) -> "ElasticCalibration":
        """Fit a calibration from elastic points.

        Each point's spectrum is Lorentzian-fit for its centre pixel (port of
        ``tender_tools.fit_peaks``); the ``(centre_pixel, mono_energy)`` pairs are
        then linear-fit for ``m``/``b`` (port of ``fit_energy_pixel``).

        Parameters
        ----------
        fit_window:
            If given, fit the Lorentzian only within ``+/- fit_window`` pixels of
            each spectrum's argmax (helps when a spectrum has structure away from
            the elastic line). ``None`` fits the whole profile.

        Raises
        ------
        ValueError
            If fewer than two elastic points are supplied (a line needs two).
        """
        if len(points) < 2:
            raise ValueError(
                f"Need >= 2 elastic points to fit a pixel->energy line; "
                f"got {len(points)}"
            )
        centers = np.array([_fit_center(p.spectrum, fit_window) for p in points])
        energies = np.array([p.energy for p in points], dtype=float)

        (m, b), _ = curve_fit(_linear, centers, energies)
        predicted = _linear(centers, m, b)
        residuals = energies - predicted
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        return cls(m=float(m), b=float(b), centers=centers, energies=energies,
                   residuals=residuals, rms=rms,
                   paths=[p.path for p in points])

    # -- apply ------------------------------------------------------------

    def to_energy(self, pixels) -> np.ndarray:
        """Convert pixel index/array to energy (eV): ``m*pixel + b``.

        Ported from ``tender_tools.make_energy_axis``.
        """
        return _linear(np.asarray(pixels, dtype=float), self.m, self.b)

    # -- persistence ------------------------------------------------------

    def save_json(self, path: str) -> str:
        """Write the calibration (coefficients + diagnostics) to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "m": self.m,
            "b": self.b,
            "rms": self.rms,
            "centers": self.centers.tolist(),
            "energies": self.energies.tolist(),
            "residuals": self.residuals.tolist(),
            "paths": [os.path.basename(p) for p in self.paths],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str) -> "ElasticCalibration":
        """Load a calibration previously written by :meth:`save_json`."""
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            m=float(d["m"]), b=float(d["b"]),
            centers=np.asarray(d.get("centers", []), dtype=float),
            energies=np.asarray(d.get("energies", []), dtype=float),
            residuals=np.asarray(d.get("residuals", []), dtype=float),
            rms=float(d.get("rms", float("nan"))),
            paths=list(d.get("paths", [])),
        )

    def __repr__(self) -> str:
        return (f"ElasticCalibration(m={self.m:.6g}, b={self.b:.6g}, "
                f"rms={self.rms:.4g} eV, n={len(self.centers)})")


def _fit_center(spectrum: np.ndarray, fit_window: int | None = None) -> float:
    """Lorentzian-fit an elastic peak and return its centre pixel.

    Ported from ``tender_tools.fit_peaks`` (which fits over pixel index). Falls
    back to the argmax pixel if the fit fails to converge.
    """
    y = np.asarray(spectrum, dtype=float)
    x = np.arange(y.size, dtype=float)
    peak = int(np.argmax(y))
    if fit_window is not None:
        lo = max(0, peak - fit_window)
        hi = min(y.size, peak + fit_window + 1)
        x, y = x[lo:hi], y[lo:hi]
    baseline = float(np.median(y))
    amp0 = float(y.max() - baseline) or float(y.max())
    p0 = [amp0, float(peak), 3.0]
    try:
        popt, _ = curve_fit(_lorentzian, x, y, p0=p0, maxfev=10000)
        return float(popt[1])
    except (RuntimeError, ValueError):
        return float(peak)


def calibrate_from_directory(directory: str, *, recursive: bool = False,
                             threshold=None, fit_window: int | None = None,
                             verbose: bool = False) -> ElasticCalibration:
    """Convenience: :func:`index_elastic` + :meth:`ElasticCalibration.fit`.

    Finds the elastic ``.sif`` files under ``directory`` and returns a fitted
    :class:`ElasticCalibration`. ``threshold`` / ``verbose`` are forwarded to the
    per-file :class:`~onepot.pipeline.OnePot` extraction; ``fit_window`` to the
    Lorentzian fit. Raises :class:`ValueError` if fewer than two elastic points
    are found.
    """
    points = index_elastic(directory, recursive=recursive, threshold=threshold,
                           verbose=verbose)
    if len(points) < 2:
        raise ValueError(
            f"Found {len(points)} elastic scan(s) in {directory!r}; need >= 2 "
            f"(files matching *elastic*.sif with a resolvable mono energy)."
        )
    return ElasticCalibration.fit(points, fit_window=fit_window)
