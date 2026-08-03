"""Detector "banana"-shape (curvature) correction.

Port of MATLAB ``sifAutoCorrelation.m``.  The emission line is not perfectly
straight across the dispersive axis: its position along the 512 spatial rows
drifts as a smooth (roughly quadratic) function of dispersive pixel.  We recover
that drift by cross-correlation and shift each dispersive column back into
alignment.

Everything here works in the package's ``(height=512, width=2048)`` convention
(see :mod:`onepot.sif_io`), which is exactly MATLAB's ``sig`` *after* its
leading ``sig'`` transpose -- so the algorithm maps across directly without the
MATLAB double-transpose bookkeeping.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, find_peaks


def _matlab_smooth(y: np.ndarray, span: int) -> np.ndarray:
    """Replicate MATLAB ``smooth(y, span)`` with the default 'moving' method.

    ``span`` is forced to the nearest odd value; interior points are a centered
    moving average, and the window shrinks symmetrically toward each end.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if span % 2 == 0:
        span -= 1
    span = max(1, min(span, n if n % 2 == 1 else n - 1))
    if span <= 1:
        return y.copy()

    out = np.empty_like(y)
    csum = np.cumsum(np.insert(y, 0, 0.0))
    half = (span - 1) // 2
    for i in range(n):
        k = min(i, n - 1 - i, half)  # symmetric shrinking window
        lo, hi = i - k, i + k + 1
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


class CurvatureCorrection:
    """Fit and apply the dispersive-axis curvature of an emission image.

    Parameters
    ----------
    t:
        Polynomial coefficients (highest degree first) of offset-vs-dispersive
        pixel.  If ``None`` (default) they are fitted from the image passed to
        :meth:`fit`.  The sentinel ``t=1`` (matching MATLAB) makes the transform
        an identity pass-through.
    n:
        Dispersive binning factor.  ``None`` uses MATLAB's default
        ``round(2**6 * min(shape) / 2048)``.
    """

    def __init__(self, t=None, n=None):
        self.t = t
        self.n = n
        self._identity = np.isscalar(t) and t == 1

    # -- fitting ----------------------------------------------------------

    def fit(self, image: np.ndarray) -> "CurvatureCorrection":
        """Estimate curvature coefficients ``self.t`` from ``image``.

        No-op when coefficients were already supplied (or ``t=1``).
        """
        if self._identity or self.t is not None:
            return self

        sig = np.asarray(image, dtype=float)  # (rows=512, dispersive=2048)
        n_rows, n_disp = sig.shape

        if self.n is None:
            n = int(round(2 ** 6 * min(sig.shape) / 2048))
        else:
            n = 2 ** self.n
        n = max(1, n)
        smooth_n = n

        nbins = n_disp // n
        # Per-bin spatial profiles (over the 512 rows), baseline-subtracted.
        odp = np.zeros((n_rows, nbins))
        for i in range(nbins):
            chunk = sig[:, i * n:(i + 1) * n].sum(axis=1)
            chunk = _matlab_smooth(chunk, smooth_n)
            odp[:, i] = chunk - chunk.min()

        sum_odp = odp.sum(axis=1)

        # Cross-correlate each bin's profile against the summed reference.
        odp2 = np.zeros((2 * n_rows - 1, nbins))
        locs = []
        for i in range(nbins):
            xc = correlate(sum_odp, odp[:, i], mode="full")
            odp2[:, i] = xc
            peaks, _ = find_peaks(xc)
            locs.append(peaks)

        center = int(np.argmax(odp2.sum(axis=1)))

        offsets = np.zeros(nbins)
        for i in range(nbins):
            if locs[i].size == 0:
                offsets[i] = 0.0
                continue
            idx = np.argmin(np.abs(locs[i] - center))
            offsets[i] = center - locs[i][idx]

        # Skip low-intensity bins, mirroring index_x in the MATLAB source.
        col_max = odp.max(axis=0)
        keep = col_max > np.median(col_max) / 10
        if keep.sum() < 3:
            keep = np.ones(nbins, dtype=bool)

        bin_centers = (np.flatnonzero(keep) + 1) * n - n / 2  # 1-based like MATLAB
        self.t = np.polyfit(bin_centers, offsets[keep], 2)
        return self

    # -- applying ---------------------------------------------------------

    def apply(self, image: np.ndarray):
        """Return ``(corrected, X)`` for ``image``.

        ``corrected`` is the straightened image (same shape); ``X`` maps each
        output pixel back to its original row index (0 where shifted in from
        outside the frame).
        """
        sig = np.asarray(image, dtype=float)
        n_rows, n_disp = sig.shape

        if self._identity:
            # MATLAB returns the untouched signal and a trivial index grid.
            X = np.tile(np.arange(1, n_rows + 1)[:, None], (1, n_disp))
            return sig.copy(), X

        if self.t is None:
            raise RuntimeError("CurvatureCorrection.apply called before fit()")

        corrected = np.zeros_like(sig)
        X = np.zeros_like(sig)
        rows = np.arange(1, n_rows + 1)  # 1-based row labels, as in MATLAB

        for j in range(n_disp):
            offset = int(round(np.polyval(self.t, j + 1)))  # +1: 1-based column
            if offset > 0:
                corrected[:n_rows - offset, j] = sig[offset:, j]
                X[:n_rows - offset, j] = rows[offset:]
            elif offset < 0:
                corrected[-offset:, j] = sig[:n_rows + offset, j]
                X[-offset:, j] = rows[:n_rows + offset]
            else:
                corrected[:, j] = sig[:, j]
                X[:, j] = rows

        return corrected, X

    def fit_apply(self, image: np.ndarray):
        """Convenience: :meth:`fit` then :meth:`apply` on the same image."""
        return self.fit(image).apply(image)
