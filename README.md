# Tender Analysis — Python SIF XES/RIXS analysis

Object-oriented Python rewrite of the MATLAB Tender x-ray analysis package (`MATLAB/sifFiles/onepot.m`, `onepotRIXS.m` and their dependencies).
Orginally written by Tsu-Chien Weng and Stanislaw Nowak.<sup>1,2</sup>
1. Abraham, B. et al. A high-throughput energy-dispersive tender X-ray spectrometer for shot-to-shot sulfur measurements. J Synchrotron Rad 26, 629–634 (2019). DOI: 10.1107/S1600577519002431
2. Nowak, S. H. et al. A versatile Johansson-type tender x-ray emission spectrometer. Review of Scientific Instruments 91, 033101 (2020). DOI: 10.1063/1.5121853

The example notebook `Tender_Analysis_Example.ipynb` walks through reading a SIF
file, the `OnePot` XES pipeline, ADU-threshold diagnostics, the `OnePotRIXS`
HERFD/XAS workflow, and batch-processing a whole beamtime directory with
`index_beamtime`. Two sample datasets are bundled under `data/`:

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
| `dataset.py`    | (new)                   | `index_beamtime()`: group a directory into runnable `Measurement`s and batch-run them |

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
    index_beamtime,
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
locates the emission-line centre by a gaussian fit.

### Batch: a whole beamtime directory

`index_beamtime` parses the `.sif` filenames in a directory and groups them into
`Measurement` objects — one XES measurement per incident energy, one RIXS
measurement per energy series — auto-pairing `*_dark.sif` files as the
background. Calibration/alignment and operando-echem files are skipped by
default and reported in `.skipped` (never silently dropped).

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
instead of aborting the batch.

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
