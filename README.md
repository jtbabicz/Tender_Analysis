# Onepot — Python SIF XES/RIXS analysis

Object-oriented Python rewrite of the MATLAB Tender x-ray analysis package (`MATLAB/sifFiles/onepot.m`, `onepotRIXS.m` and their dependencies). 
Orginally written by Tsu-Chien Weng and Stanislaw Nowak.<sup>1,2</sup>   
1. Abraham, B. et al. A high-throughput energy-dispersive tender X-ray spectrometer for shot-to-shot sulfur measurements. J Synchrotron Rad 26, 629–634 (2019). DOI: 10.1107/S1600577519002431
2. Nowak, S. H. et al. A versatile Johansson-type tender x-ray emission spectrometer. Review of Scientific Instruments 91, 033101 (2020). DOI: 10.1063/1.5121853
   
An example jupyter notebooks is included `Tender_Analysis_Example.ipynb`, along with a Na<sub>2</sub>SO<sub>4</sub> sulfur RIXS example dataset. 

## Package layout

| Module          | Ports (MATLAB)          | Purpose                                              |
|-----------------|-------------------------|------------------------------------------------------|
| `sif_io.py`     | `sifread.m` + `sif_get_*`| `SifFile`: read frames via `sif_parser`, parse `I0`/`I1`/`mono` from the comment |
| `files.py`      | `sifFindFiles.m`        | `find_sif_files()`: glob + natural sort              |
| `common.py`     | (shared)                | ADU histogram + common-mode (zero-peak) estimation   |
| `background.py` | `sifBatchBackground.m`  | `compute_background()`: min-projection background    |
| `analyze.py`    | `sifAnalyze.m`          | `extract_signal()`: single-photon event extraction   |
| `curvature.py`  | `sifAutoCorrelation.m`  | `CurvatureCorrection`: banana-shape fit + apply      |
| `pipeline.py`   | `onepot.m`, `onepotRIXS.m`| `OnePot` / `OnePotRIXS` orchestrators              |

## Orientation convention

Frames are `(n_frames, height=512, width=2048)`. The 2048-pixel width is the
energy-dispersive axis; a spectrum is `frame.sum(axis=0)` (length 2048). This
collapses the MATLAB double-transpose into one documented convention — see
`sif_io.py`.

## Usage

```python
from onepot import OnePot, OnePotRIXS, SifFile

# XES: extract single-photon signal, using a dark frame as background
dark = SifFile("scan_dark.sif").frame(0)
result = OnePot("emission_2524.00eV_01.sif", bcg=dark).run()
spectrum = result.spectrum()          # length-2048 emission spectrum

# RIXS/HERFD: build a scan map over many energies, extract an emission band
rixs = OnePotRIXS("scan_*.sif").herfd(n=3, i0_corr=True)
rixs.E, rixs.HERFD, rixs.TFY          # incident energy, HERFD line-out, TFY
```

`bcg=None` computes the background from the data; `bcg=0` disables it; passing an
array uses it directly. `evolution=True` runs the two-pass curvature workflow.

## Dependencies

numpy, scipy, natsort, pytest, sif_parser

## Setup & tests

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy sif_parser natsort pytest
.venv/bin/python -m pytest tender_rewrite/tests -v
```

Tests run against the two sample files in `../siftests/`. A full RIXS energy scan
needs many files, so the RIXS test is a single-energy smoke check.
