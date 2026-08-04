"""Discovering and ordering ``.sif`` files.

Port of MATLAB ``sifFindFiles.m``: expand a wildcard path (or accept an explicit
list) into a naturally-sorted list of ``.sif`` paths.  Natural sort keeps
``foo_2.sif`` ahead of ``foo_10.sif`` (MATLAB used ``sort_nat``); we use the
``natsort`` package for the same effect.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable

from natsort import natsorted


def find_sif_files(files: str | Iterable[str], file_nbrs=None) -> list[str]:
    """Return a naturally-sorted list of ``.sif`` paths.

    Parameters
    ----------
    files:
        A glob pattern (``"scan_*.sif"``), a single path, or an iterable of
        explicit paths.  A pattern with no ``.sif`` suffix has ``*.sif`` behaviour
        applied so ``"scan_"`` matches ``scan_*.sif``.
    file_nbrs:
        Optional 0-based indices selecting a subset of the discovered files.
        ``None`` / empty means "all".

    Raises
    ------
    FileNotFoundError
        If a pattern matches nothing.
    """
    # Explicit list/tuple of paths -> use as given (still natural-sorted).
    if not isinstance(files, str):
        flist = [str(f) for f in files]
    else:
        pattern = files
        if not glob.has_magic(pattern):
            # A bare path or prefix: exact file, or prefix -> prefix*.sif.
            if pattern.endswith(".sif") and os.path.isfile(pattern):
                flist = [pattern]
            else:
                stem = pattern[:-4] if pattern.endswith(".sif") else pattern
                flist = glob.glob(stem + "*.sif")
        else:
            flist = glob.glob(pattern)
            # Keep only .sif matches, mirroring sifFindFiles' filtering.
            flist = [f for f in flist if f.endswith(".sif")]

    flist = natsorted(flist)

    if not flist:
        raise FileNotFoundError(f"No .sif files matched: {files!r}")

    if _is_set(file_nbrs):
        idx = _as_indices(file_nbrs)
        try:
            flist = [flist[i] for i in idx]  # 0-based indices
        except IndexError as exc:
            raise IndexError(f"file_nbrs {file_nbrs} out of range for {len(flist)} files") from exc

    return flist


def _is_set(file_nbrs) -> bool:
    if file_nbrs is None:
        return False
    if isinstance(file_nbrs, int):
        return True
    return len(file_nbrs) > 0


def _as_indices(file_nbrs) -> list[int]:
    if isinstance(file_nbrs, int):
        return [file_nbrs]
    return list(file_nbrs)
