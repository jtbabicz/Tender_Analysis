"""Python rewrite of the MATLAB SIF XES/RIXS analysis.

Public API mirrors the two MATLAB entry points:

- :class:`OnePot`      -- single-file / scan XES analysis (``onepot.m``)
- :class:`OnePotRIXS`  -- HERFD/XAS extraction from a scan map (``onepotRIXS.m``)

Supporting pieces (:class:`SifFile`, :func:`find_sif_files`,
:func:`compute_background`, :func:`extract_signal`, :class:`CurvatureCorrection`)
are also exported for direct use.
"""

from .sif_io import SifFile
from .files import find_sif_files
from .background import compute_background
from .curvature import CurvatureCorrection
from .analyze import extract_signal, XESResult
from .pipeline import OnePot, OnePotRIXS, Thresholds, RIXSResult
from .dataset import (
    index_beamtime, Measurement, FileRecord, BeamtimeIndex, parse_sif_name,
    MeasurementRun, run_measurement,
)

__all__ = [
    "OnePot",
    "OnePotRIXS",
    "Thresholds",
    "XESResult",
    "RIXSResult",
    "SifFile",
    "find_sif_files",
    "compute_background",
    "extract_signal",
    "CurvatureCorrection",
    "index_beamtime",
    "Measurement",
    "FileRecord",
    "BeamtimeIndex",
    "parse_sif_name",
    "MeasurementRun",
    "run_measurement",
]
