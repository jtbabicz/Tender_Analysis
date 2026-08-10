"""Sample-directory indexing and measurement grouping.

A directory here (e.g. ``CPMoITriCO3Dimer/``) typically holds all measurements
for a *single compound* collected within a beamtime -- many compounds are
measured per beamtime, each in its own directory. (The public functions retain
the ``beamtime`` name for backward compatibility, but they operate on one
sample's directory.) Such a directory holds many *measurements* mixed together:
XES emission scans (fixed incident energy, one or more scan indices, each with a
paired dark) and RIXS scans (a full incident-energy series with one dark). The
:class:`OnePot` / :class:`OnePotRIXS` pipelines each analyze a single
measurement, so this module supplies the missing layer: parse the ``.sif``
filenames into structured records and group them into runnable
:class:`Measurement` objects, each of which instantiates the right pipeline
(with auto-paired darks wired in as the background).

The filename parser (:func:`parse_sif_name` and its token tables) is adapted
from ``tender/build_report.py::parse_name`` -- the reference parser that has
classified thousands of datasets across every tender beamtime. Only the parts
needed to *run* analysis are copied here (line/technique/energy/dark/aux
detection); the element-inference and compound-stem logic from that script is
intentionally left out. If the beamline naming conventions evolve, reconcile
the two.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from natsort import natsorted

from .pipeline import OnePot, OnePotRIXS
from .sif_io import SifFile

# -- filename grammar (lifted from tender/build_report.py) ----------------

# Emission-line tokens, longest first so ``Ka12`` beats ``Ka``.
LINE_TOKENS = [
    "L3_val", "L3val", "L2val", "Ka12",
    "Ka", "Kb", "La", "Lb", "Ma", "Mb",
]
LINE_RE = re.compile(
    r"(?:^|[_\-])([A-Z][a-z]?)?(" + "|".join(LINE_TOKENS) + r")(?=RIXS|XES|HERFD|[_\-.]|$)"
)

# Auxiliary / calibration keywords (plain substring match on the lowercased core).
AUX_KEYWORDS = ["elastic", "alignment", "align", "background", "bkg",
                "test", "pgm", "calib", "reference_grid", "focus", "belowedge"]

# Elastic-scattering scans (pixel->energy calibration). These are a *subset* of
# the aux files above (``elastic`` is an AUX_KEYWORD), so index_beamtime still
# skips them from the standard workflow; the calibration module (see
# ``calibration.py``) opts back in by filtering on ``FileRecord.is_elastic``.
_ELASTIC_RE = re.compile(r"elastic", re.IGNORECASE)

# Commissioning / spectrometer-alignment patterns (case-insensitive regex).
AUX_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"rowland", r"(?:^|_)dy_?scan", r"(?:^|_)dy[+\-]\d", r"(?:^|_)dx[+\-]\d",
        r"(?:^|_)example\d", r"(?:^|_)windows?_\d+deg", r"noshutter",
        r"(?:^|_)slit\d", r"m0slit",
    )
]

# Operando echem naming (non-standard / unstandardized -> skipped by default).
ECHEM_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"electrolyte", r"hclo4", r"h2so4", r"(?:^|[_\-])ocv",
        r"[_\-]\d?[pP]?\d*[Vv](?=_|$)",       # applied potential 0p4V 1P2V
        r"(?:^|[_\-])E\d{1,2}(?=_|$)",        # electrode / cell id E2..E6
        r"echem",
    )
]

ENERGY_RE = re.compile(r"(\d{4}(?:\.\d+)?)")
_EV_RE = re.compile(r"(\d{4}(?:\.\d+)?)\s*eV", re.IGNORECASE)
_TECH_RE = re.compile(r"(RIXS|RXES|HERFD|XES)", re.IGNORECASE)
# Trailing scan / series indices in the two schemas.
_XES_SCAN_RE = re.compile(r"_(\d{1,3})$")               # ..._<energy>eV_05
_RIXS_SERIES_RE = re.compile(r"_RIXS_(\d{1,3})(?=_|$)", re.IGNORECASE)

# When no technique token is present, treat a group spanning this many distinct
# incident energies as an energy scan (RIXS). Mirrors build_report.py.
_RIXS_ENERGY_THRESHOLD = 5


@dataclass
class FileRecord:
    """One parsed ``.sif`` filename."""

    path: str
    sample: str                       # sample + conditions label
    emission_line: str | None         # e.g. "Ka", "L3val" (None if absent)
    technique_token: str | None       # explicit RIXS/XES/HERFD token, if named
    energy: float | None              # incident energy (eV), if found
    scan_index: int | None            # trailing XES scan index
    series_index: int | None          # RIXS series index
    is_dark: bool
    is_aux: bool
    is_echem: bool
    is_elastic: bool = False          # elastic-scattering calibration scan
    kind: str | None = None           # "XES"/"RIXS", set by _assign_kinds


def parse_sif_name(path: str) -> FileRecord | None:
    """Decode one ``.sif`` path into a :class:`FileRecord` (``None`` if not a sif).

    Adapted from ``tender/build_report.py::parse_name``. The sample label is the
    filename with the energy / technique / emission-line / dark tokens stripped
    out; if nothing is left it falls back to the parent directory name.
    """
    name = os.path.basename(path)
    if not name.lower().endswith(".sif"):
        return None
    base = name[:-4]
    low = base.lower()

    is_dark = bool(re.search(r"_?dark$", low))
    core = re.sub(r"_?dark$", "", base, flags=re.IGNORECASE)
    low_core = core.lower()
    is_aux = (any(k in low_core for k in AUX_KEYWORDS)
              or any(p.search(core) for p in AUX_PATTERNS))
    is_echem = any(p.search(core) for p in ECHEM_PATTERNS)
    is_elastic = bool(_ELASTIC_RE.search(core))

    # explicit technique token, if present
    if "rixs" in low_core or "rxes" in low_core:
        technique_token = "RIXS"
    elif "herfd" in low_core:
        technique_token = "HERFD"
    elif "xes" in low_core:
        technique_token = "XES"
    else:
        technique_token = None

    # RIXS series index (from the ``_RIXS_<n>_`` token)
    ms = _RIXS_SERIES_RE.search(core)
    series_index = int(ms.group(1)) if ms else None

    # emission line (ignore an optional element prefix, which we don't need here)
    m = LINE_RE.search(core)
    emission_line = _norm_line(m.group(2)) if m else None

    # incident energy: prefer an ``eV``-tagged value, else an in-range 4-digit
    # token. Scan from the END: a leading ``YYMMDD`` date (e.g. ``250615`` ->
    # ``2506``) can fall in range, but the real (RIXS) energy sits at the tail.
    # (Latent edge: a *trailing* 4-digit date/timestamp after the energy in an
    # un-eV-tagged name would be picked instead; no such convention exists today.)
    energy = None
    mev = _EV_RE.search(core)
    if mev:
        energy = float(mev.group(1))
    else:
        for e in reversed(ENERGY_RE.findall(core)):
            f = float(e)
            if 1900 < f < 4300:
                energy = f
                break

    # sample + conditions label: strip energy / technique / line tokens.
    # Remove the ``_RIXS_<n>_`` series token as a unit *before* the generic
    # technique strip, so the series number isn't left orphaned in the label.
    label = core
    label = _EV_RE.sub("", label)
    # Strip the series token with the SAME boundary-guarded pattern used for
    # detection (`_RIXS_SERIES_RE`, which requires ``_``/end after the digits).
    # A plain ``_RIXS_\d{1,3}`` would eat the leading digits of a bare energy in
    # names like ``..._RIXS_2825.00.sif`` (RIXS with no series index).
    label = _RIXS_SERIES_RE.sub("", label)
    label = _TECH_RE.sub("", label)
    if m:
        # Strip the emission-line token from ``label`` directly; do NOT slice with
        # ``m.start()``/``m.end()`` -- those offsets index into ``core``, but
        # ``label`` has already been shortened above, so slicing corrupts it.
        label = LINE_RE.sub("_", label, count=1)
    label = re.sub(r"_\d{4}(\.\d+)?(?=_|$)", "", label)   # leftover bare energy
    label = re.sub(r"[_\-]{2,}", "_", label).strip("_- ")

    # XES trailing scan index (only meaningful once the label is cleaned up, and
    # only for XES -- a RIXS file's trailing index is its series, handled above)
    scan_index = None
    if series_index is None:
        msc = _XES_SCAN_RE.search(label)
        if msc:
            scan_index = int(msc.group(1))
            label = _XES_SCAN_RE.sub("", label).strip("_- ")

    if not label or re.fullmatch(r"\d+", label):
        dirbase = os.path.basename(os.path.dirname(path))
        dirbase = re.sub(r"(?i)_(RXES|RIXS|XES|HERFD)$", "", dirbase)
        # Guarantee a non-empty label so unrelated files never collapse under
        # ``sample==''``: prefer the (cleaned) directory name, else the filename.
        label = dirbase or label or base

    return FileRecord(
        path=path, sample=label, emission_line=emission_line,
        technique_token=technique_token, energy=energy, scan_index=scan_index,
        series_index=series_index, is_dark=is_dark, is_aux=is_aux,
        is_echem=is_echem, is_elastic=is_elastic,
    )


def _norm_line(tok: str) -> str:
    return {"L3_val": "L3val"}.get(tok, tok)


def _as_list(paths) -> list[str]:
    """Normalize a save_txt return (str or list) to a list of paths."""
    return [paths] if isinstance(paths, str) else list(paths)


@dataclass
class Measurement:
    """A group of ``.sif`` files that make up one runnable analysis.

    ``kind`` is ``"XES"`` (fixed incident energy, scans summed) or ``"RIXS"``
    (an incident-energy series). ``data_paths`` are the signal files and
    ``dark_paths`` the auto-paired dark(s) used as the background.
    """

    kind: str
    sample: str
    emission_line: str | None
    incident_energy: float | None      # XES: fixed value; RIXS: None (a series)
    series_index: int | None
    data_paths: list[str] = field(default_factory=list)
    dark_paths: list[str] = field(default_factory=list)

    # -- construction of the underlying pipeline --------------------------

    def pipeline(self, **overrides):
        """Return a configured :class:`OnePot` / :class:`OnePotRIXS`.

        Darks are wired in automatically: XES passes the frame-averaged dark as
        ``bcg``; RIXS uses ``use_dark_as_background=True``. Any keyword override
        (``threshold``, ``bcg``, ...) takes precedence.
        """
        if self.kind == "RIXS":
            # Hand every file (data + dark) to OnePotRIXS; it separates the dark
            # itself. Prefer the dark as background when exactly one is present.
            kw = dict(use_dark_as_background=(len(self.dark_paths) == 1))
            kw.update(overrides)
            if "bcg" in kw and kw["bcg"] is not None:
                kw.pop("use_dark_as_background", None)  # explicit bcg wins
            return OnePotRIXS(self.data_paths + self.dark_paths, **kw)

        kw = dict(overrides)
        if "bcg" not in kw and self.dark_paths:
            kw["bcg"] = self._dark_background()
        return OnePot(self.data_paths, **kw)

    #: HERFD/RIXS-only keywords, consumed by :meth:`OnePotRIXS.herfd`.
    _HERFD_KEYS = ("central_pix", "n", "i0_corr")

    def run(self, **overrides):
        """Instantiate the pipeline and execute it.

        XES returns an ``XESResult`` (from ``OnePot.run``); RIXS returns a
        ``RIXSResult`` (from ``OnePotRIXS.herfd``). ``herfd`` keywords
        (``central_pix``, ``n``, ``i0_corr``) apply to RIXS; they are split out
        here and *ignored* for XES, so a single ``run_all(central_pix=...)`` over
        a mixed beamtime doesn't blow up the XES measurements.
        """
        herfd_kw = {k: overrides.pop(k) for k in self._HERFD_KEYS if k in overrides}
        if self.kind == "RIXS":
            return self.pipeline(**overrides).herfd(**herfd_kw)
        return self.pipeline(**overrides).run()

    def _dark_background(self) -> np.ndarray:
        """Frame-averaged mean of the paired dark file(s).

        NOTE: for XES this averages *all* darks paired to the measurement (e.g.
        one dark per scan index at this incident energy) into a single background
        image. This assumes the dark is stationary across the scans. Revisit
        per-scan dark pairing if a systematic drift appears between early and
        late scans (S/N change, beamline drift) -- see ``_group_xes``.
        """
        frames = [SifFile(p).data.mean(axis=0) for p in self.dark_paths]
        return np.mean(frames, axis=0)

    # -- naming for exports ----------------------------------------------

    def label(self) -> str:
        """Filesystem-safe stem from the measurement metadata."""
        parts = [self.sample, self.emission_line or "", self.kind]
        if self.kind == "XES" and self.incident_energy is not None:
            parts.append(f"{self.incident_energy:g}eV")
        elif self.kind == "RIXS" and self.series_index is not None:
            parts.append(f"{self.series_index:02d}")
        stem = "_".join(p for p in parts if p)
        return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")

    def save_result(self, result, root: str, *, subdir: str = "analysis",
                    **save_kwargs) -> list[str]:
        """Auto-name ``result`` into ``root`` and write it via ``result.save_txt``.

        The filename is built from this measurement's metadata (see :meth:`label`)
        and written under ``root/<subdir>/`` (set ``subdir=""`` to write directly
        into ``root``). Use this for whole-beamtime batch runs; for a one-off
        explicit path call ``result.save_txt(path)`` directly. Returns the paths
        written.
        """
        out_dir = os.path.join(root, subdir) if subdir else root
        path = os.path.join(out_dir, self.label() + ".txt")
        # Each result's ``save_txt`` accepts a different set of knobs; drop the
        # ones the target doesn't understand so a single batch call with mixed
        # measurement kinds doesn't raise. ``save_map`` is RIXS-only;
        # ``calibration`` (pixel->emission-energy axis) is XES-only.
        if self.kind == "RIXS":
            save_kwargs = {k: v for k, v in save_kwargs.items()
                           if k != "calibration"}
        else:
            save_kwargs = {k: v for k, v in save_kwargs.items() if k != "save_map"}
        return _as_list(result.save_txt(path, **save_kwargs))

    def __repr__(self) -> str:
        e = (f"{self.incident_energy:g}eV" if self.incident_energy is not None
             else f"series {self.series_index}")
        return (f"Measurement({self.kind}, {self.sample!r}, "
                f"{self.emission_line}, {e}, "
                f"{len(self.data_paths)} files, {len(self.dark_paths)} dark)")


@dataclass
class MeasurementRun:
    """Outcome of running one :class:`Measurement`.

    ``result`` is the ``XESResult`` / ``RIXSResult`` on success (``None`` on
    failure, or when ``keep_result=False`` dropped it after saving); ``error``
    holds the exception when a measurement raised; ``seconds`` is the wall-clock
    analysis time; ``summary`` holds a few small scalar outputs (spectrum sum,
    peak pixel, ...) that survive even when the heavy arrays are dropped.
    """

    measurement: Measurement
    result: object | None = None
    seconds: float = 0.0
    error: Exception | None = None
    saved_paths: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def _result_summary(m: Measurement, result) -> dict:
    """Small, picklable scalar summary of a result (survives array-dropping)."""
    out = {"kind": m.kind}
    try:
        if m.kind == "RIXS":
            out["central_pix"] = int(getattr(result, "central_pix", -1))
            herfd = getattr(result, "HERFD", None)
            if herfd is not None:
                out["herfd_sum"] = float(np.nansum(herfd))
        else:
            spec = result.spectrum()
            out["spectrum_sum"] = float(spec.sum())
            out["peak_pixel"] = int(np.argmax(spec))
    except Exception:  # noqa: BLE001 -- summary is best-effort, never fatal
        pass
    return out


def run_measurement(m: Measurement, *, save_root: str | None = None,
                    save_kwargs: dict | None = None, keep_result: bool = True,
                    **overrides) -> MeasurementRun:
    """Run a single measurement and wrap the outcome in a :class:`MeasurementRun`.

    This is the one unit of work behind :meth:`BeamtimeIndex.run_all` -- a
    standalone, picklable function so it can be dispatched to a process pool
    without touching callers. Exceptions are captured on the returned record
    rather than raised, so one bad measurement doesn't abort a batch.

    ``keep_result=False`` drops the heavy result arrays after saving (keeping only
    ``saved_paths`` + a small ``summary``); this is used for parallel runs, where
    a full ``XESResult`` (~37 MB) would otherwise be pickled back to the parent.
    """
    t0 = time.perf_counter()
    try:
        result = m.run(**overrides)
        run = MeasurementRun(measurement=m, result=result,
                             seconds=time.perf_counter() - t0,
                             summary=_result_summary(m, result))
        if save_root is not None:
            run.saved_paths = m.save_result(result, root=save_root,
                                            **(save_kwargs or {}))
        if not keep_result:
            run.result = None  # drop heavy arrays; summary/saved_paths remain
        return run
    except Exception as exc:  # noqa: BLE001 -- batch robustness, re-surfaced on the record
        return MeasurementRun(measurement=m, error=exc,
                              seconds=time.perf_counter() - t0)


def _merge_overrides(m: Measurement, param_fn, overrides: dict) -> dict:
    """Overlay a per-measurement resolver on top of the global overrides.

    ``param_fn`` (if given) is called with the :class:`Measurement` and returns a
    dict of pipeline overrides that take precedence over the shared ``overrides``
    for *this* measurement only. Returning ``None`` / ``{}`` means "use the
    globals unchanged". With ``param_fn=None`` the global ``overrides`` are passed
    through verbatim, so a single global ``threshold=[...]`` still applies to
    every measurement (backward compatible).

    Resolution happens in the *parent* process, so ``param_fn`` itself never
    crosses the process boundary in parallel mode -- each worker still receives a
    plain, picklable dict (the merged dict must be picklable, same as today).
    """
    if param_fn is None:
        return overrides
    extra = param_fn(m) or {}
    if not isinstance(extra, dict):
        raise TypeError(
            f"param_fn must return a dict (or None); got {type(extra).__name__} "
            f"for measurement {m.label()!r}"
        )
    return {**overrides, **extra}


def _resolve_workers(max_workers: int | None) -> int:
    """Resolve the worker count, failing safe when the allocation is unknown.

    Order: an explicit ``max_workers`` always wins; else a reliable SLURM
    allocation signal (``SLURM_CPUS_PER_TASK``, then ``SLURM_CPUS_ON_NODE``);
    else raise. We deliberately do NOT fall back to ``os.cpu_count()`` /
    ``sched_getaffinity`` -- on a shared login/compute node those report the
    whole machine and would oversubscribe cores belonging to other users/jobs.
    """
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        return max_workers
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    raise RuntimeError(
        "Cannot determine a safe worker count: no explicit max_workers and no "
        "SLURM allocation (SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE) detected. "
        "Pass max_workers=N explicitly (matched to your allocation) to avoid "
        "oversubscribing a shared node."
    )


def _manifest_entry(run: "MeasurementRun") -> dict:
    """Flat, JSON-serializable record of one measurement run."""
    m = run.measurement
    return {
        "label": m.label(),
        "kind": m.kind,
        "sample": m.sample,
        "emission_line": m.emission_line,
        "incident_energy": m.incident_energy,
        "series_index": m.series_index,
        "n_data_files": len(m.data_paths),
        "n_dark_files": len(m.dark_paths),
        "ok": run.ok,
        "saved_paths": list(run.saved_paths),
        "seconds": round(run.seconds, 3),
        "summary": run.summary,
        "error": None if run.ok else f"{type(run.error).__name__}: {run.error}",
    }


def _write_manifest(path: str, runs: list["MeasurementRun"]) -> str:
    """Write a JSON run manifest (per-measurement records + a small summary)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(runs)
    n_ok = sum(r.ok for r in runs)
    manifest = {
        "n": n,
        "n_ok": n_ok,
        "n_failed": n - n_ok,
        "total_seconds": round(sum(r.seconds for r in runs), 3),
        "measurements": [_manifest_entry(r) for r in runs],
    }
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _print_run_line(run: "MeasurementRun", i: int, n: int, verbose: bool) -> None:
    """Print one batch progress line for a completed run (no-op if not verbose)."""
    if not verbose:
        return
    label = run.measurement.label()
    if run.ok:
        saved = f"  -> {len(run.saved_paths)} file(s)" if run.saved_paths else ""
        print(f"  [{i}/{n}] {label:52s} {run.seconds:6.1f}s  ok{saved}")
    else:
        print(f"  [{i}/{n}] {label:52s} {run.seconds:6.1f}s  "
              f"FAILED: {type(run.error).__name__}: {run.error}")


@dataclass
class BeamtimeIndex:
    """Result of :func:`index_beamtime`: measurements plus a skipped-file bucket.

    Indexes one sample directory (typically a single compound's measurements
    collected within a beamtime; the ``Beamtime`` name is kept for backward
    compatibility).
    """

    measurements: list[Measurement]
    skipped: dict[str, list[str]] = field(default_factory=dict)  # reason -> paths

    def __iter__(self):
        return iter(self.measurements)

    def __len__(self):
        return len(self.measurements)

    def by_kind(self, kind: str) -> list[Measurement]:
        return [m for m in self.measurements if m.kind == kind]

    def run_all(self, *, save_root: str | None = None, save_kwargs: dict | None = None,
                verbose: bool = True, detail: bool = False,
                max_workers: int | None = None, manifest: bool = True,
                param_fn=None, **overrides) -> list[MeasurementRun]:
        """Run every measurement in the index and return a list of outcomes.

        Parameters
        ----------
        save_root:
            If given, each result is auto-named and written under
            ``save_root/analysis/`` via :meth:`Measurement.save_result`, and a
            JSON run manifest is written to ``save_root/analysis/run_manifest.json``
            (unless ``manifest=False``). Required when running in parallel.
        save_kwargs:
            Extra keywords forwarded to ``save_result`` (e.g. ``save_map=True``,
            or ``calibration=<ElasticCalibration>`` to write an energy axis).
        verbose:
            Print a per-measurement progress line (name, time, status) and a
            closing summary (default ``True``).
        detail:
            Also stream each measurement's own progress (header + per-file lines)
            by running it with ``verbose=True`` (default ``False``). Ignored in
            parallel mode (child stdout would interleave across processes).
        max_workers:
            ``None`` / ``1`` -> run sequentially (default; full results kept on
            each record). ``>1`` (or, when the caller passes ``max_workers`` but
            wants the allocation resolved, see :func:`_resolve_workers`) -> run
            measurements concurrently in a process pool. Parallel mode **requires**
            ``save_root`` (results are saved in the worker and the heavy arrays are
            dropped before returning, so nothing is lost); results come back in
            completion order, not index order.
        manifest:
            Write the JSON run manifest when ``save_root`` is given (default True).
        param_fn:
            Optional resolver ``param_fn(measurement) -> dict | None``. The global
            ``**overrides`` are the baseline applied to *every* measurement; the
            dict returned here overlays per-measurement adjustments on top for that
            one measurement (e.g. a different ``threshold`` for a concentrated vs.
            a dilute sample). Returning ``None`` / ``{}`` keeps the globals. With
            ``param_fn=None`` (default) behavior is identical to passing globals
            alone. The callable runs in the parent process, so it need not be
            picklable even in parallel mode (see :func:`_merge_overrides`).
        **overrides:
            Pipeline overrides passed through to :meth:`Measurement.run`
            (e.g. ``threshold=[100, 170, 350]``, ``scan_nbrs=range(20)``,
            ``central_pix=1280`` for RIXS). Must be picklable in parallel mode.

        Notes
        -----
        A failure is captured on its :class:`MeasurementRun` (``error``) instead
        of aborting the batch. Each run is dispatched through
        :func:`run_measurement`, the single unit of work.
        """
        n = len(self.measurements)
        parallel = max_workers is not None and (max_workers != 1)
        # A bare max_workers=1 is explicitly sequential; anything else parallel.
        if parallel:
            workers = _resolve_workers(max_workers)
            if save_root is None:
                raise ValueError(
                    "Parallel run_all (max_workers>1) requires save_root: results "
                    "are saved in the worker and their arrays dropped before "
                    "returning, so they must be written to disk."
                )
            runs = self._run_parallel(workers, save_root, save_kwargs, verbose,
                                      param_fn, overrides)
        else:
            if detail:
                overrides.setdefault("verbose", True)
            runs = self._run_sequential(save_root, save_kwargs, verbose,
                                        param_fn, overrides)

        if verbose:
            n_ok = sum(r.ok for r in runs)
            total = sum(r.seconds for r in runs)
            print(f"Done: {n_ok}/{n} succeeded in {total:.1f}s"
                  + (f" ({n - n_ok} failed)" if n_ok < n else ""))
        if save_root is not None and manifest:
            path = _write_manifest(
                os.path.join(save_root, "analysis", "run_manifest.json"), runs)
            if verbose:
                print(f"Wrote manifest: {path}")
        return runs

    def _run_sequential(self, save_root, save_kwargs, verbose, param_fn, overrides):
        runs: list[MeasurementRun] = []
        n = len(self.measurements)
        if verbose:
            print(f"Running {n} measurement(s)...")
        for i, m in enumerate(self.measurements, 1):
            merged = _merge_overrides(m, param_fn, overrides)
            run = run_measurement(m, save_root=save_root, save_kwargs=save_kwargs,
                                  **merged)
            runs.append(run)
            _print_run_line(run, i, n, verbose)
        return runs

    def _run_parallel(self, workers, save_root, save_kwargs, verbose, param_fn,
                      overrides):
        runs: list[MeasurementRun] = []
        n = len(self.measurements)
        overrides.pop("verbose", None)  # child stdout can't interleave cleanly
        if verbose:
            print(f"Running {n} measurement(s) on {workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Resolve param_fn in the parent so only plain dicts cross to workers.
            futures = [pool.submit(run_measurement, m, save_root=save_root,
                                   save_kwargs=save_kwargs, keep_result=False,
                                   **_merge_overrides(m, param_fn, overrides))
                       for m in self.measurements]
            for done, fut in enumerate(as_completed(futures), 1):
                run = fut.result()
                runs.append(run)
                _print_run_line(run, done, n, verbose)
        return runs


def index_beamtime(directory: str, *, skip_echem: bool = True,
                   skip_aux: bool = True, recursive: bool = False) -> BeamtimeIndex:
    """Index a sample directory into runnable :class:`Measurement` groups.

    The directory typically holds all measurements for a single compound
    collected within a beamtime (the ``beamtime`` name is retained for backward
    compatibility).

    Parameters
    ----------
    directory:
        Folder holding ``.sif`` files (or a glob pattern).
    skip_echem:
        Drop operando-echem measurements (non-standard naming) (default True).
    skip_aux:
        Drop calibration / alignment / test files (default True).
    recursive:
        Recurse into subdirectories (default False).

    Returns
    -------
    BeamtimeIndex
        ``.measurements`` are the grouped runnable units; ``.skipped`` buckets
        the files that were not grouped (keys: ``"aux"``, ``"echem"``,
        ``"unparsed"``, ``"no_energy"``) so nothing is silently dropped.
    """
    if glob.has_magic(directory):
        paths = glob.glob(directory, recursive=recursive)
    elif recursive:
        paths = glob.glob(os.path.join(directory, "**", "*.sif"), recursive=True)
    else:
        paths = glob.glob(os.path.join(directory, "*.sif"))

    records: list[FileRecord] = []
    skipped: dict[str, list[str]] = {}

    def _skip(reason: str, path: str) -> None:
        skipped.setdefault(reason, []).append(path)

    # Natural sort (foo_2 before foo_10), matching find_sif_files / the rest of
    # the package rather than plain lexical ordering.
    for p in natsorted(paths):
        rec = parse_sif_name(p)
        if rec is None:
            _skip("unparsed", p)
            continue
        if skip_aux and rec.is_aux:
            _skip("aux", p)
            continue
        if skip_echem and rec.is_echem:
            _skip("echem", p)
            continue
        records.append(rec)

    darks = [r for r in records if r.is_dark]
    data = [r for r in records if not r.is_dark]

    _assign_kinds(data)  # sets rec.kind on each record
    measurements: list[Measurement] = []
    measurements += _group_rixs([r for r in data if r.kind == "RIXS"], darks)
    measurements += _group_xes([r for r in data if r.kind == "XES"], darks, _skip)

    measurements.sort(key=lambda m: (m.kind, m.sample, m.emission_line or "",
                                     m.incident_energy or 0,
                                     m.series_index or 0))
    return BeamtimeIndex(measurements=measurements, skipped=skipped)


def _assign_kinds(data: list[FileRecord]) -> None:
    """Set ``rec.kind`` (``"XES"``/``"RIXS"``) on each record in place.

    Explicit technique token wins. Otherwise a ``(sample, emission_line)`` group
    spanning >= ``_RIXS_ENERGY_THRESHOLD`` distinct incident energies is an
    energy scan (RIXS); anything else is XES.
    """
    from collections import defaultdict

    groups: dict[tuple, list[FileRecord]] = defaultdict(list)
    for r in data:
        groups[(r.sample, r.emission_line)].append(r)

    for grp in groups.values():
        untok = [r for r in grp if r.technique_token is None]
        distinct_e = {r.energy for r in untok if r.energy is not None}
        scan = len(distinct_e) >= _RIXS_ENERGY_THRESHOLD
        for r in grp:
            if r.technique_token in ("RIXS", "HERFD"):
                r.kind = "RIXS"
            elif r.technique_token == "XES":
                r.kind = "XES"
            else:
                r.kind = "RIXS" if scan else "XES"


def _group_rixs(data: list[FileRecord],
                darks: list[FileRecord]) -> list[Measurement]:
    """Group RIXS records into one measurement per (sample, line, series)."""
    from collections import defaultdict

    groups: dict[tuple, list[FileRecord]] = defaultdict(list)
    for r in data:
        groups[(r.sample, r.emission_line, r.series_index)].append(r)

    out = []
    for (sample, line, series), recs in groups.items():
        dpaths = [d.path for d in darks
                  if d.sample == sample and d.emission_line == line
                  and d.series_index == series]
        out.append(Measurement(
            kind="RIXS", sample=sample, emission_line=line,
            incident_energy=None, series_index=series,
            data_paths=natsorted(r.path for r in recs), dark_paths=natsorted(dpaths),
        ))
    return out


def _group_xes(data: list[FileRecord], darks: list[FileRecord],
               skip) -> list[Measurement]:
    """Group XES records into one measurement per (sample, line, energy)."""
    from collections import defaultdict

    groups: dict[tuple, list[FileRecord]] = defaultdict(list)
    for r in data:
        if r.energy is None:
            skip("no_energy", r.path)
            continue
        groups[(r.sample, r.emission_line, round(r.energy, 2))].append(r)

    out = []
    for (sample, line, energy), recs in groups.items():
        # Pair darks by matching sample/line/energy; fall back to scan-index match.
        # All darks matched here are collected together and averaged into one
        # background image by Measurement._dark_background (i.e. per-scan darks
        # are NOT kept separate). This is fine when the dark is stationary; if a
        # systematic scan-to-scan change shows up, switch to per-scan pairing.
        scan_idxs = {r.scan_index for r in recs}
        dpaths = [d.path for d in darks
                  if d.sample == sample and d.emission_line == line
                  and (d.energy is not None and round(d.energy, 2) == energy)]
        if not dpaths:
            dpaths = [d.path for d in darks
                      if d.sample == sample and d.emission_line == line
                      and d.scan_index in scan_idxs]
        out.append(Measurement(
            kind="XES", sample=sample, emission_line=line,
            incident_energy=energy, series_index=None,
            data_paths=natsorted(r.path for r in recs), dark_paths=natsorted(dpaths),
        ))
    return out


def _beamtime_subdir(directory: str) -> str:
    """Output subdirectory name for one input directory: ``<parent>_<name>``.

    Prefixing with the parent folder disambiguates the common case where the same
    compound is measured in more than one beamtime and stored in identically-named
    directories (e.g. ``2025-06/CPMoTriCO3Dimer`` and ``2026-02/CPMoTriCO3Dimer``):
    without the prefix both would write to the same output tree and silently
    overwrite each other. Falls back to just ``<name>`` when there is no parent
    component (a bare name or filesystem root).
    """
    norm = os.path.normpath(directory)
    name = os.path.basename(norm)
    parent = os.path.basename(os.path.dirname(norm))
    stem = f"{parent}_{name}" if parent else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")


def run_beamtimes(directories, *, save_root: str, max_workers: int | None = None,
                  index_kwargs: dict | None = None, verbose: bool = True,
                  param_fn=None, **overrides) -> dict[str, list[MeasurementRun]]:
    """Index and run many sample directories, one output tree per directory.

    Each directory typically holds one compound's measurements collected within a
    beamtime (the ``beamtime`` name is kept for backward compatibility). Thin
    composable driver over :func:`index_beamtime` + :meth:`BeamtimeIndex.run_all`
    for automated re-processing of many past directories. Each directory is
    indexed and run with its outputs written under
    ``save_root/<parent>_<directory_name>/analysis/`` (its own per-directory
    manifest included); a combined ``save_root/run_manifest.json`` is written
    across all directories. The output name is prefixed with the parent folder so
    the same compound measured in two beamtimes (identically-named directories,
    e.g. ``2025-06/Sample`` and ``2026-02/Sample``) writes to distinct trees
    instead of overwriting; if two inputs still resolve to the same output name a
    :class:`ValueError` is raised rather than silently clobbering.

    Parameters mirror :meth:`BeamtimeIndex.run_all` (``max_workers``, ``param_fn``,
    ``**overrides`` such as ``threshold``); ``index_kwargs`` is forwarded to
    :func:`index_beamtime` (e.g. ``{"skip_echem": False}``). ``save_root`` is
    required. Because ``param_fn`` receives each :class:`Measurement` (which
    carries ``sample``, ``emission_line``, ``incident_energy``, ``kind``, and
    ``label()``), one resolver can key off the sample to vary parameters per
    measurement across every directory. Returns ``{directory: [MeasurementRun, ...]}``.

    This is deliberately just a loop over the public pieces -- for full control
    (or a SLURM job array with one directory per task) call ``index_beamtime`` +
    ``run_all`` yourself.
    """
    index_kwargs = index_kwargs or {}
    all_runs: dict[str, list[MeasurementRun]] = {}
    flat: list[MeasurementRun] = []
    seen: dict[str, str] = {}   # output subdir -> source directory (collision guard)
    for directory in directories:
        name = _beamtime_subdir(directory)
        if name in seen and seen[name] != directory:
            raise ValueError(
                f"Output name {name!r} maps to two different input directories "
                f"({seen[name]!r} and {directory!r}); their results would overwrite "
                f"each other under {save_root!r}. Rename one input directory or run "
                f"them with separate save_root values."
            )
        seen[name] = directory
        if verbose:
            print(f"=== beamtime: {name} ===")
        idx = index_beamtime(directory, **index_kwargs)
        runs = idx.run_all(save_root=os.path.join(save_root, name),
                           max_workers=max_workers, verbose=verbose,
                           param_fn=param_fn, **overrides)
        all_runs[directory] = runs
        flat.extend(runs)

    path = _write_manifest(os.path.join(save_root, "run_manifest.json"), flat)
    if verbose:
        n_ok = sum(r.ok for r in flat)
        print(f"All beamtimes: {n_ok}/{len(flat)} measurements ok. "
              f"Combined manifest: {path}")
    return all_runs
