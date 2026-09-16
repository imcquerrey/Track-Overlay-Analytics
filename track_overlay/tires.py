"""Tire temperature, pressure, and icon rendering."""

import numpy as np

from .drawing import FONT, LINE_DRAW, FS, S, TH, put_text, rect_fill, rect_outline, text_size
from .settings import (
    FRONT_TIRE_TEMP_COLD_F, FRONT_TIRE_TEMP_GREEN_HIGH_F,
    FRONT_TIRE_TEMP_GREEN_LOW_F, FRONT_TIRE_TEMP_HOT_F,
    FRONT_TIRE_TEMP_YELLOW_HIGH_F, FRONT_TIRE_TEMP_YELLOW_LOW_F,
    REAR_TIRE_TEMP_COLD_F, REAR_TIRE_TEMP_GREEN_HIGH_F,
    REAR_TIRE_TEMP_GREEN_LOW_F, REAR_TIRE_TEMP_HOT_F,
    REAR_TIRE_TEMP_YELLOW_HIGH_F, REAR_TIRE_TEMP_YELLOW_LOW_F,
)
from .telemetry import kelvin_to_f, kpa_to_psi, psi_abs_to_psig


def tire_temp_to_color(temp_f, axle="front"):
    if not np.isfinite(temp_f):
        return (120, 120, 120)

    if axle.lower().startswith("r"):
        cold = float(REAR_TIRE_TEMP_COLD_F)
        g_lo = float(REAR_TIRE_TEMP_GREEN_LOW_F)
        g_hi = float(REAR_TIRE_TEMP_GREEN_HIGH_F)
        y_lo = float(REAR_TIRE_TEMP_YELLOW_LOW_F)
        y_hi = float(REAR_TIRE_TEMP_YELLOW_HIGH_F)
        hot = float(REAR_TIRE_TEMP_HOT_F)
    else:
        cold = float(FRONT_TIRE_TEMP_COLD_F)
        g_lo = float(FRONT_TIRE_TEMP_GREEN_LOW_F)
        g_hi = float(FRONT_TIRE_TEMP_GREEN_HIGH_F)
        y_lo = float(FRONT_TIRE_TEMP_YELLOW_LOW_F)
        y_hi = float(FRONT_TIRE_TEMP_YELLOW_HIGH_F)
        hot = float(FRONT_TIRE_TEMP_HOT_F)

    g_lo = max(g_lo, cold)
    g_hi = max(g_hi, g_lo + 1e-6)
    y_lo = max(y_lo, g_hi)
    y_hi = max(y_hi, y_lo + 1e-6)
    hot = max(hot, y_hi + 1e-6)

    if temp_f <= g_lo:
        denom = max(g_lo - cold, 1e-6)
        r = float(np.clip((temp_f - cold) / denom, 0.0, 1.0))
        b = int(255 * (1 - r))
        g = int(255 * r)
        return (b, g, 0)

    if temp_f <= g_hi:
        return (0, 255, 0)

    if temp_f < y_lo:
        denom = max(y_lo - g_hi, 1e-6)
        r = float(np.clip((temp_f - g_hi) / denom, 0.0, 1.0))
        rr = int(255 * r)
        return (0, 255, rr)

    if temp_f <= y_hi:
        return (0, 255, 255)

    denom = max(hot - y_hi, 1e-6)
    r = float(np.clip((temp_f - y_hi) / denom, 0.0, 1.0))
    g = int(255 * (1 - r))
    rr = 255
    return (0, g, rr)


def draw_tire_icon(frame, cx, cy, w, h, fill_bgr, psi_text):
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    rect_fill(frame, x1, y1, x2, y2, fill_bgr)
    rect_outline(frame, x1, y1, x2, y2, (255, 255, 255), 1)
    # w/h already scaled; don't scale inset again
    inset = int(0.20 * min(w, h))
    rect_fill(frame, x1 + inset, y1 + inset, x2 -
              inset, y2 - inset, (25, 25, 25))
    (tw, th), _ = text_size(psi_text, FONT, FS(0.55), TH(2))
    put_text(frame, psi_text, (int(cx - tw / 2), int(cy + th / 2)), FONT, FS(0.55), (255, 255, 255), TH(2),
             LINE_DRAW)


def draw_tires(frame, vals, x, y, scale_temp, scale_tire_p):
    tw, th = S(76), S(120)
    gap_x, gap_y = S(22), S(18)
    grid = [
        ("t_fl", "p_fl", x, y),
        ("t_fr", "p_fr", x + tw + gap_x, y),
        ("t_rl", "p_rl", x, y + th + gap_y),
        ("t_rr", "p_rr", x + tw + gap_x, y + th + gap_y)
    ]
    for tk, pk, px, py in grid:
        t_raw = vals.get(tk, np.nan)
        p_raw = vals.get(pk, np.nan)
        t_k = (t_raw / scale_temp) if np.isfinite(t_raw) else np.nan
        t_f = kelvin_to_f(t_k) if np.isfinite(t_k) else np.nan
        color = tire_temp_to_color(t_f, axle=(
            'rear' if tk in ('t_rl', 't_rr') else 'front'))
        p_kpa_abs = (p_raw / scale_tire_p) if np.isfinite(p_raw) else np.nan
        p_psi_abs = kpa_to_psi(p_kpa_abs) if np.isfinite(p_kpa_abs) else np.nan
        p_psig = psi_abs_to_psig(p_psi_abs) if np.isfinite(
            p_psi_abs) else np.nan
        if np.isfinite(p_psig):
            p_psig = max(0.0, p_psig)
        psi_text = f"{p_psig:0.1f}" if np.isfinite(p_psig) else "--"
        draw_tire_icon(frame, px + tw / 2, py + th /
                       2, tw, th, color, psi_text)
