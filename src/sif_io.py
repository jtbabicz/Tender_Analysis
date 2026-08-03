"""Reading Andor ``.sif`` detector files.

This wraps :func:`sif_parser.np_open` (the same backend used in
``siftests/sifTests.ipynb``) and adds a parser for the acquisition comment
block, which holds the ``I0``/``I1``/``mono`` values used downstream by the RIXS
analysis.  ``sif_parser`` does not expose that comment, so we read it straight
from the file header bytes -- this mirrors the regex extraction done by the
MATLAB helpers ``sif_get_I0.m`` / ``sif_get_I1.m`` / ``sif_get_mono.m``.

Orientation convention
-----------------------
``sif_parser`` returns image data as ``(n_frames, height, width)`` -- for these
detectors ``(n_frames, 512, 2048)``.  The 2048-pixel ``width`` axis is the
energy-dispersive direction; the 512-row ``height`` axis is summed to produce a
spectrum.  Every frame in this package is therefore a ``(height, width)`` =
``(512, 2048)`` array, and a spectrum is ``frame.sum(axis=0)`` giving a
length-2048 vector -- matching the notebook's ``np.sum(frame, axis=0)``.

The MATLAB code carried an extra transpose (``sifread`` returned
``(Height, Width)`` then transposed, and ``sifAutoCorrelation`` transposed
again).  We collapse all of that into this single documented convention so the
rest of the package never has to think about it.
"""

from __future__ import annotations

import re
from functools import cached_property

import numpy as np
import sif_parser

# Keys parsed out of the acquisition comment block. Values look like
# ``mono = 2524.0000`` / ``I0 = 338821`` in the raw header text.
_COMMENT_KEYS = ("mono", "I0", "I1", "exptime", "numkins", "vert")

# The comment sits within the first few kB of the file, ahead of the binary
# image payload. Reading a generous slice avoids decoding the pixel data.
_HEADER_BYTES = 8192


class SifFile:
    """A single Andor ``.sif`` file: image frames plus metadata.

    Parameters
    ----------
    path:
        Filesystem path to the ``.sif`` file.

    Notes
    -----
    Frame data is loaded lazily on first access and cached.  See the module
    docstring for the ``(n_frames, 512, 2048)`` orientation convention.
    """

    def __init__(self, path: str):
        self.path = str(path)

    @cached_property
    def _loaded(self):
        data, info = sif_parser.np_open(self.path)
        return np.asarray(data, dtype=float), info

    @property
    def data(self) -> np.ndarray:
        """All frames as a ``(n_frames, height, width)`` float array."""
        return self._loaded[0]

    @property
    def info(self) -> dict:
        """Raw metadata dict from :func:`sif_parser.np_open`."""
        return self._loaded[1]

    @property
    def num_frames(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of a single frame."""
        return self.data.shape[1], self.data.shape[2]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def exposure_time(self) -> float:
        return float(self.info.get("ExposureTime", 0.0))

    def frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` (0-based) as a ``(height, width)`` array."""
        return self.data[index]

    def spectrum(self, index: int | None = None) -> np.ndarray:
        """Sum over detector rows to get a length-``width`` spectrum.

        With ``index=None`` all frames are summed first.
        """
        if index is None:
            return self.data.sum(axis=(0, 1))
        return self.data[index].sum(axis=0)

    # -- comment metadata -------------------------------------------------

    @cached_property
    def comment(self) -> str:
        """The acquisition comment block as text (best-effort decode)."""
        with open(self.path, "rb") as fh:
            head = fh.read(_HEADER_BYTES)
        return head.decode("latin-1", errors="replace")

    @cached_property
    def metadata(self) -> dict[str, float]:
        """Numeric values parsed from the comment (``mono``, ``I0`` ...).

        Missing keys map to ``nan`` so callers can test with ``math.isnan``.
        """
        out: dict[str, float] = {}
        for key in _COMMENT_KEYS:
            m = re.search(rf"{re.escape(key)}\s*=\s*(-?\d+\.?\d*)", self.comment)
            out[key] = float(m.group(1)) if m else float("nan")
        return out

    @property
    def mono(self) -> float:
        """Monochromator (incident) energy from the comment."""
        return self.metadata["mono"]

    @property
    def I0(self) -> float:
        """Incident-beam intensity monitor from the comment."""
        return self.metadata["I0"]

    @property
    def I1(self) -> float:
        """Transmitted-beam intensity monitor from the comment."""
        return self.metadata["I1"]

    def __repr__(self) -> str:
        return f"SifFile({self.path!r}, frames={self.num_frames})"
