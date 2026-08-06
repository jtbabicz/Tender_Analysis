# Tender Analysis — Python SIF XES/RIXS analysis

Object-oriented Python rewrite of the MATLAB Tender x-ray analysis package (`MATLAB/sifFiles/onepot.m`, `onepotRIXS.m` and their dependencies).
Orginally written by Tsu-Chien Weng and Stanislaw Nowak.<sup>1,2</sup>
1. Abraham, B. et al. A high-throughput energy-dispersive tender X-ray spectrometer for shot-to-shot sulfur measurements. J Synchrotron Rad 26, 629–634 (2019). DOI: 10.1107/S1600577519002431
2. Nowak, S. H. et al. A versatile Johansson-type tender x-ray emission spectrometer. Review of Scientific Instruments 91, 033101 (2020). DOI: 10.1063/1.5121853

The example notebook `Tender_Analysis_Example.ipynb` walks through reading a SIF
file, the `OnePot` XES pipeline, ADU-threshold diagnostics, the `OnePotRIXS`
HERFD/XAS workflow, and batch-processing a whole sample directory with
`index_beamtime` (including parallel execution and a JSON run manifest). A
directory such as `data/CPMoITriCO3Dimer/` holds all the measurements for a
*single compound* collected within a beamtime — many compounds are measured per
beamtime, each in its own directory. (The batch functions keep the historical
`beamtime` name for backward compatibility, but they operate on one such
sample directory.) Two sample datasets are bundled under `data/`:

- `data/Na2SO4/` — sulfur K RIXS energy scan (Na<sub>2</sub>SO<sub>4</sub> pellet).
- `data/CPMoITriCO3Dimer/` — Mo L<sub>3</sub> valence-to-core XES of the
  \[CpMo(CO)<sub>3</sub>\]<sub>2</sub> dimer.

## Package layout

The package lives in `src/`. Modules port the MATLAB routines one-to-one:

| Module          | Ports (MATLAB)          | Purpose                                              |
|-----------------|-------------------------|------------------------------------------------------|
| `sif_io.py`     | `sifread.m` + `sif_get_*`| `SifFile`: read frames via `sif_parser`, parse `I0`/`I1`/`mono` from the comment |
| `files.py`      | `sifFindFiles.m`        | `find_sif_files()`: glob + natural sort              |
| `common.py`     | (shared)                | ADU histogram + common-mode (zero-peak) estimation   |
| `background.py` | `sifBatchBackground.m`  | `compute_background()`: min-projection background    |
| `analyze.py`    | `sifAnalyze.m`          | `extract_signal()`: single-photon event extraction   |
| `curvature.py`  | `sifAutoCorrelation.m`  | `CurvatureCorrection`: banana-shape fit + apply      |
| `pipeline.py`   | `onepot.m`, `onepotRIXS.m`| `OnePot` / `OnePotRIXS` orchestrators              |
| `dataset.py`    | (new)                   | `index_beamtime()` / `run_beamtimes()`: group a sample directory into runnable `Measurement`s, batch-run them (optionally parallel, with per-measurement parameters), and write a JSON manifest |
| `calibration.py`| (new)                   | `ElasticCalibration` / `calibrate_from_directory()`: fit a pixel→energy calibration from elastic scans (post-beamtime, optional) |

## Orientation convention

Frames are `(n_frames, height=512, width=2048)`. The 2048-pixel width is the
energy-dispersive axis; a spectrum is `frame.sum(axis=0)` (length 2048). This
collapses the MATLAB double-transpose into one documented convention — see
`sif_io.py`.

## Usage

There are two levels of entry point:

- **`OnePot` / `OnePotRIXS`** — analyze a *single* measurement from an explicit
  set of files (you choose the background and options).
- **`index_beamtime`** — scan a *directory*, group the files into measurements,
  and run them in batch. It builds `OnePot` / `OnePotRIXS` under the hood, so it
  layers on top of the single-measurement API rather than replacing it.

### Single measurement

The public API is exported from the `src` package (as imported in the notebook):

```python
from src import (
    SifFile, find_sif_files, compute_background,
    extract_signal, CurvatureCorrection, OnePot, OnePotRIXS,
    index_beamtime, run_beamtimes,
)

# Read a single SIF file
sif = SifFile("Na2SO4_pellet_20pcSucrose_SKa_RIXS_01_2465.00.sif")
sif.I0, sif.mono, sif.num_frames, sif.shape   # metadata parsed from the comment
frame = sif.frame(0)                           # (512, 2048) image
spectrum = frame.sum(axis=0)                   # length-2048 emission spectrum

# XES: extract single-photon signal from a scan (glob, path, or file list)
op = OnePot("data/CPMoITriCO3Dimer/*MoL3val_2523.00eV_0*.sif",
            bcg=None, threshold=[100, 170, 350], histograms=True)
xes = op.run()
spec = xes.spectrum()                          # length-2048 emission spectrum

# RIXS/HERFD: build a scan map over many energies, extract an emission band
rixs = OnePotRIXS("data/Na2SO4/*.sif")
out = rixs.herfd(central_pix=1280, n=7, i0_corr=True)
out.E, out.HERFD, out.TFY, out.rixs_map        # incident energy, line-outs, map
```

`bcg=None` computes the background from the data; `bcg=0` disables it; passing an
array uses it directly. `evolution=True` runs the two-pass curvature workflow.
`OnePotRIXS` excludes `*_dark.sif` frames from the scan by default
(`exclude_dark=False` keeps them; `use_dark_as_background=True` subtracts the
averaged dark instead of a min-projection background). `herfd(central_pix=None)`
locates the emission-line centre by a gaussian fit. `verbose=True` prints a
progress header and a per-file line during extraction (most of a measurement's
time is spent there).

### Batch: a whole sample directory

`index_beamtime` parses the `.sif` filenames in a directory (typically one
compound's measurements collected within a beamtime) and groups them into
`Measurement` objects — one XES measurement per incident energy, one RIXS
measurement per energy series — auto-pairing `*_dark.sif` files as the
background. Calibration/alignment (including elastic scans) and operando-echem
files are skipped by default and reported in `.skipped` (never silently dropped).

```python
idx = index_beamtime("data/CPMoITriCO3Dimer")   # -> BeamtimeIndex
len(idx), idx.skipped                            # measurement count + skipped files
for m in idx:
    print(m)                                     # sample, line, technique, energy

# Run one measurement (dark auto-paired as background); overrides pass through
# to the underlying pipeline.
result = idx.by_kind("XES")[0].run(threshold=[100, 170, 350])

# Or run the whole directory in one call, saving each result to text.
runs = idx.run_all(save_root="data/CPMoITriCO3Dimer",
                   threshold=[100, 170, 350])     # prints progress + summary
ok = [r for r in runs if r.ok]                    # MeasurementRun: .result/.seconds/.error
```

`run_all(**overrides)` forwards options to each pipeline by keyword (order does
not matter); a measurement that raises is captured on its `MeasurementRun.error`
instead of aborting the batch. Pass `detail=True` to also stream each
measurement's own per-file progress. When `save_root` is given, `run_all` also
writes a JSON run manifest to `save_root/analysis/run_manifest.json` (one entry
per measurement: label, ok/failed, output paths, timing, error) so unattended
batch runs are auditable.

#### Per-measurement parameters

The `**overrides` above are a single baseline applied to *every* measurement.
When measurements in the same directory need different parameters — e.g. a
concentrated and a dilute version of a sample need different ADU thresholds —
pass `param_fn`, a resolver `param_fn(measurement) -> dict` whose returned dict
is overlaid **on top of** the global overrides for that one measurement only:

```python
def per_measurement(m):
    # m carries .sample, .emission_line, .incident_energy, .kind, .label()
    if "dilute" in m.sample:
        return {"threshold": [60, 110, 250]}   # override just this one
    return {}                                  # keep the global baseline

runs = idx.run_all(save_root="out/CPMoITriCO3Dimer",
                   threshold=[100, 170, 350],   # baseline for everything
                   param_fn=per_measurement)
```

Returning `{}` / `None` keeps the global values, so with `param_fn=None`
(default) behavior is identical to passing globals alone. `run_beamtimes` takes
the same `param_fn` — since the resolver keys off the `Measurement`, one function
can span every directory. The resolver runs in the parent process, so it need
not be picklable even in parallel mode (only the resulting overrides dict does,
as before). If reorganizing the files into separate directories is simpler than
writing a resolver, that remains a valid alternative — `param_fn` is here for
when it is not.

### Parallel batch execution (HPC / many beamtimes)

For large re-processing jobs, `run_all(max_workers=N)` runs measurements
concurrently in a process pool (each measurement is an independent unit of
work). Parallel mode requires `save_root`: results are written in the worker and
their heavy arrays dropped before returning, so the batch avoids shipping large
arrays back between processes.

```python
# Parallel over a directory (must set save_root).
runs = idx.run_all(save_root="out/CPMoITriCO3Dimer",
                   max_workers=8, threshold=[100, 170, 350])

# Many sample directories in one call -> per-directory outputs + combined manifest.
run_beamtimes(["beamtimes/2025-06/SampleA", "beamtimes/2025-06/SampleB"],
              save_root="out", max_workers=8, threshold=[100, 170, 350])
```

Worker-count resolution is deliberately conservative for shared clusters: an
explicit `max_workers` always wins; otherwise it reads the SLURM allocation
(`SLURM_CPUS_PER_TASK`, then `SLURM_CPUS_ON_NODE`); if neither is available it
raises rather than guess from the machine's total core count (which would
oversubscribe a shared node). Because the pool uses the *spawn* start method, a
plain script must guard its entry point:

```python
if __name__ == "__main__":
    run_beamtimes([...], save_root="out", max_workers=8)
```

### Exporting results

Both result objects write annotated text (a commented header with source files,
thresholds, and background mode, followed by data columns):

```python
result.save_txt("mo_2523.txt")                    # XES: pixel, counts
rixs_out.save_txt("na2so4.txt", save_map=True)     # RIXS: energy, HERFD, TFY (+ _map)
```

For batch runs, `Measurement.save_result(result, root=...)` auto-names the file
from the measurement metadata into an `analysis/` subdirectory (this is what
`run_all(save_root=...)` calls).

### Energy calibration (pixel → energy)

The energy-dispersive (pixel) axis is calibrated by collecting elastic
scattering at several fixed monochromator energies: each elastic peak lands on a
particular pixel, so fitting the peak centres and linear-fitting
`(centre_pixel → mono_energy)` gives the `energy = m·pixel + b` conversion.

This is a **standalone, skippable** step — elastic data is often unavailable
until post-beamtime analysis, so it is never built or required by the standard
workflow. Elastic `.sif` files (`*elastic*.sif`) are skipped by
`index_beamtime` as before; the calibration API opts back in explicitly:

```python
from src import calibrate_from_directory, ElasticCalibration

# Fit from a directory of elastic scans, then persist for reuse.
cal = calibrate_from_directory("data/elastic_2024-06")   # -> ElasticCalibration
cal.m, cal.b, cal.rms                                    # coeffs + RMS residual (eV)
cal.save_json("calib.json")
cal = ElasticCalibration.load_json("calib.json")         # reload later

# Apply it at export time: XES spectra gain an `energy_eV` column
# (pixel, energy_eV, counts). Without a calibration, output is unchanged.
result.save_txt("mo_2523.txt", calibration=cal)
runs = idx.run_all(save_root="out", threshold=[100, 170, 350],
                   save_kwargs={"calibration": cal})      # applied to XES results
```

`save_kwargs={"calibration": cal}` is dropped for RIXS results (whose export
already carries an incident-energy axis), so a mixed-kind batch is safe.

## Dependencies

numpy, scipy, natsort, matplotlib, sif_parser (plus pytest for tests)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib sif_parser natsort pytest
```

Then launch the example notebook:

```bash
.venv/bin/jupyter lab Tender_Analysis_Example.ipynb
```
