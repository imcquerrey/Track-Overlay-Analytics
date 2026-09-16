"""Signal synchronization and minimap projection mathematics."""

import numpy as np


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s < 1e-9:
        return x * 0.0
    return (x - m) / s


def _resample_interp(t_src, y_src, t_dst):
    # np.interp requires increasing t
    return np.interp(t_dst, t_src, y_src, left=y_src[0], right=y_src[-1])


def _match_speed_window_fft(speed_big, speed_small):
    # Returns best start index in speed_big where speed_small matches.
    a = _zscore(speed_big)
    b = _zscore(speed_small)
    n = len(b)
    m = len(a)
    if n < 10 or m < n + 10:
        return 0

    # Convolution for sliding dot product: dot(i) = sum a[i:i+n] * b
    # Use FFT of a and reversed(b)
    size = _next_pow2(m + n - 1)
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b[::-1], size)
    conv = np.fft.irfft(fa * fb, size)

    # valid i corresponds to conv[i + n - 1]
    dots = conv[(n - 1):(m)]
    # Only keep valid starts: i in [0, m-n]
    dots = dots[: (m - n + 1)]
    i0 = int(np.argmax(dots))
    return i0


def _project_point_to_polyline_windowed_m(p_m, poly_m, seg_hint=None, window=120):
    """Project p_m (meters) onto polyline poly_m (Nx2 meters).
    If seg_hint is provided, restrict search to +-window segments around seg_hint (circular).
    Returns (proj_xy, seg_idx, t) where seg_idx is 0..n-2 and t is [0..1] within that segment.
    """
    p = np.asarray(p_m, dtype=np.float64)
    pts = np.asarray(poly_m, dtype=np.float64)
    n = int(len(pts))
    if n < 2:
        return (pts[0] if n == 1 else p), 0, 0.0

    seg_n = n - 1
    if seg_hint is None:
        cand = np.arange(seg_n, dtype=np.int32)
    else:
        w = int(max(5, window))
        deltas = np.arange(-w, w + 1, dtype=np.int32)
        cand = (int(seg_hint) + deltas) % seg_n

    a = pts[cand]
    b = pts[cand + 1]
    ab = b - a
    ap = p - a
    ab_len2 = np.sum(ab * ab, axis=1) + 1e-12
    t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
    proj = a + (ab * t[:, None])
    d2 = np.sum((proj - p) ** 2, axis=1)
    k = int(np.argmin(d2))
    seg_idx = int(cand[k])
    return proj[k], seg_idx, float(t[k])


def _unwrap_near(x, ref):
    """Return x (possibly +/-1) that is closest to ref."""
    cands = (x, x + 1.0, x - 1.0)
    return min(cands, key=lambda v: abs(v - ref))


def _project_point_to_polyline_m(p_m, poly_m):
    # Returns closest point on polyline to p_m (both in meters), and its index.
    p = np.asarray(p_m, dtype=np.float64)
    pts = np.asarray(poly_m, dtype=np.float64)
    if len(pts) < 2:
        return pts[0] if len(pts) == 1 else p, 0

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ap = p - a
    ab_len2 = np.sum(ab * ab, axis=1) + 1e-12
    t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
    proj = a + (ab * t[:, None])
    d2 = np.sum((proj - p) ** 2, axis=1)
    i = int(np.argmin(d2))
    return proj[i], i
