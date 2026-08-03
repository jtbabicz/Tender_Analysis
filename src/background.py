"""Common background estimation across many frames.

Port of MATLAB ``sifBatchBackground.m``.  For each frame the zero peak (common
mode) is subtracted, then the per-pixel *minimum* over all frames is retained --
this captures the fixed, non-stochastic background (e.g. photofluorescence)
while rejecting single-photon events, which land above the minimum.  Finally the
mean common mode is added back so the background sits at the true ADU level.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .common import common_mode
from .sif_io import SifFile


def compute_background(
    files: Iterable[SifFile],
    scan_nbrs=0,
    subtract_common_mode: bool = True,
) -> np.ndarray:
    """Compute the common background image from a set of SIF files.

    Parameters
    ----------
    files:
        Iterable of :class:`~onepot.sif_io.SifFile`.
    scan_nbrs:
        Optional 1-based frame indices (across the concatenated frame stream) to
        include; ``0``/empty means all frames.
    subtract_common_mode:
        Subtract each frame's zero peak before taking the minimum (default, as in
        the MATLAB code).

    Returns
    -------
    np.ndarray
        The ``(height, width)`` background image.

    Raises
    ------
    ValueError
        If fewer than two frames are available (matching MATLAB's
        "Not enough frames or sif files").
    """
    files = list(files)
    total_frames = sum(f.num_frames for f in files)
    if total_frames < 2:
        raise ValueError("Not enough frames or sif files")

    selected = _selected_frames(scan_nbrs, total_frames)

    shape = files[0].shape
    bcg = np.full(shape, np.inf)
    total_common_mode = 0.0
    n_used = 0

    counter = 0  # 1-based global frame index, matching MATLAB
    for sif in files:
        for i in range(sif.num_frames):
            counter += 1
            if counter not in selected:
                continue
            frame = sif.frame(i)
            if subtract_common_mode:
                cm = common_mode(frame, refine=False)
                total_common_mode += cm
                n_used += 1
                bcg = np.minimum(bcg, frame - cm)
            else:
                bcg = np.minimum(bcg, frame)

    if subtract_common_mode and n_used:
        bcg = bcg + total_common_mode / n_used

    return bcg


def _selected_frames(scan_nbrs, total_frames) -> set[int]:
    """Resolve ``scan_nbrs`` to a set of 1-based frame indices."""
    if scan_nbrs is None or (isinstance(scan_nbrs, int) and scan_nbrs == 0):
        return set(range(1, total_frames + 1))
    if isinstance(scan_nbrs, int):
        return {scan_nbrs}
    arr = np.asarray(list(scan_nbrs))
    if arr.dtype == bool:
        return set(np.flatnonzero(arr) + 1)
    return set(int(x) for x in arr)
