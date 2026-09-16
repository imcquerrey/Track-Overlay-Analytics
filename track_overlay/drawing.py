"""Shared scaling, text caching, compositing, gauges, and dashboard drawing."""

import cv2
import numpy as np

from .settings import FAST_NO_AA, UI_SCALE

LINE_DRAW = cv2.LINE_8 if FAST_NO_AA else cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_CACHE = {}
_TEXTSZ = {}
_PANEL_CACHE = {}


def S(px): return int(round(px * UI_SCALE))


def FS(f): return float(f * UI_SCALE)


def TH(t): return max(1, int(round(t * UI_SCALE)))


def put_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=LINE_DRAW, bottomLeftOrigin=False):
    """
    Drop-in replacement for cv2.putText with caching.
    Matches cv2.putText signature and returns img.
    org is the baseline-bottom-left point (unless bottomLeftOrigin=True).
    """
    if text is None:
        return img
    try:
        s = str(text)
    except Exception:
        return img
    if s == "":
        return img

    key = (s, fontFace, float(fontScale), tuple(int(c) for c in color), int(thickness), int(lineType),
           bool(bottomLeftOrigin))
    cached = _TEXT_CACHE.get(key)

    if cached is None:
        # compute text size
        (tw, th), baseline = cv2.getTextSize(s, fontFace, fontScale, thickness)
        pad = 4  # pixels
        w = tw + pad * 2
        h = th + baseline + pad * 2
        patch = np.zeros((h, w, 3), dtype=np.uint8)

        # draw text into patch (baseline at y = pad + th)
        x0 = pad
        y0 = pad + th
        # outline first (black) for readability (slightly thicker)
        outline_th = max(thickness + 2, 2)
        cv2.putText(patch, s, (x0, y0), fontFace, fontScale,
                    (0, 0, 0), outline_th, lineType, bottomLeftOrigin)
        cv2.putText(patch, s, (x0, y0), fontFace, fontScale,
                    color, thickness, lineType, bottomLeftOrigin)

        alpha = (patch[:, :, 0] | patch[:, :, 1] |
                 patch[:, :, 2]).astype(np.uint8)
        # make alpha 0/255
        alpha = np.where(alpha > 0, 255, 0).astype(np.uint8)
        cached = (patch, alpha, pad, th)
        _TEXT_CACHE[key] = cached

    patch, alpha, pad, th = cached

    x, y = int(org[0]), int(org[1])

    # Convert baseline org -> top-left of patch
    # baseline y = top + pad + th  => top = y - (pad + th)
    top = y - (pad + th)
    left = x - pad

    H, W = img.shape[:2]
    ph, pw = patch.shape[:2]
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(W, left + pw)
    y1 = min(H, top + ph)
    if x0 >= x1 or y0 >= y1:
        return img

    px0 = x0 - left
    py0 = y0 - top
    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)

    roi = img[y0:y1, x0:x1]
    p = patch[py0:py1, px0:px1]
    a = alpha[py0:py1, px0:px1]

    mask = a > 0
    # fast overwrite (opaque text pixels)
    roi[mask] = p[mask]
    return img


def text_size(txt, font, scale, thickness):
    k = (txt, float(scale), int(thickness))
    v = _TEXTSZ.get(k)
    if v is None:
        v = cv2.getTextSize(txt, font, scale, thickness)
        _TEXTSZ[k] = v
    return v


def alpha_blend(dst, src_rgba, x, y):
    """Fast uint8 alpha blend (no float32), for 4-channel src_rgba over BGR dst at (x,y)."""
    h, w = src_rgba.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
    if x0 >= x1 or y0 >= y1:
        return dst

    roi = dst[y0:y1, x0:x1]
    src = src_rgba[(y0 - int(y)):(y1 - int(y)), (x0 - int(x)):(x1 - int(x))]

    # uint16 math: out = (roi*(255-a) + src_rgb*a)/255
    a = src[..., 3:4].astype(np.uint16)  # 0..255
    inv = (255 - a)
    roi16 = roi.astype(np.uint16)
    src16 = src[..., :3].astype(np.uint16)
    out = (roi16 * inv + src16 * a + 127) // 255
    roi[:] = out.astype(np.uint8)
    return dst
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    src = src_rgba[(y0 - y):(y1 - y), (x0 - x):(x1 - x)].astype(np.float32)
    a = src[..., 3:4] / 255.0
    roi = roi * (1 - a) + src[..., :3] * a
    dst[y0:y1, x0:x1] = roi.astype(np.uint8)
    return dst


def draw_panel_roi(frame, x, y, w, h, draw_fn, *args, **kwargs):
    """Draw into a ROI VIEW (no copy). Fastest path: all drawing stays inside the smaller slice."""
    H, W = frame.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, x0 + int(w))
    y1 = min(H, y0 + int(h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]  # writable view
    draw_fn(roi, *args, **kwargs)


def panel_rgba(w, h, alpha=110, shade=0):
    """Small perf win: cache frequently used solid RGBA panels."""
    key = (int(w), int(h), int(alpha), int(shade))
    p = _PANEL_CACHE.get(key)
    if p is None:
        p = np.zeros((key[1], key[0], 4), dtype=np.uint8)
        p[..., 0] = key[3]
        p[..., 1] = key[3]
        p[..., 2] = key[3]
        p[..., 3] = key[2]
        _PANEL_CACHE[key] = p
    return p


def rect_fill(frame, x1, y1, x2, y2, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_8)


def rect_outline(frame, x1, y1, x2, y2, color=(255, 255, 255), thickness=1):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, TH(thickness), cv2.LINE_8)


def draw_tile(frame, x, y, w, h, label, value, value_scale=0.70):
    rect_fill(frame, x, y, x + w, y + h, (18, 18, 18))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    put_text(frame, label, (x + S(12), y + S(20)), FONT,
             FS(0.40), (210, 210, 210), TH(1), LINE_DRAW)
    put_text(frame, value, (x + S(12), y + h - S(10)), FONT,
             FS(value_scale), (255, 255, 255), TH(2), LINE_DRAW)


def draw_tile_fuel(frame, x, y, w, h, label, psi_text, pct_text):
    rect_fill(frame, x, y, x + w, y + h, (18, 18, 18))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    put_text(frame, label, (x + S(12), y + S(20)), FONT,
             FS(0.40), (210, 210, 210), TH(1), LINE_DRAW)
    if pct_text:
        (tw, _), _ = text_size(pct_text, FONT, FS(0.35), TH(1))
        put_text(frame, pct_text, (x + w - S(12) - tw, y + S(20)), FONT, FS(0.35), (210, 210, 210), TH(1),
                 LINE_DRAW)
    put_text(frame, psi_text, (x + S(12), y + h - S(10)),
             FONT, FS(0.66), (255, 255, 255), TH(2), LINE_DRAW)


def draw_bar(frame, pct, x, y, w, h, fill_bgr):
    rect_fill(frame, x, y, x + w, y + h, (12, 12, 12))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    v = float(np.clip(pct, 0, 100))
    fh = int((h - S(8)) * (v / 100.0))
    cv2.rectangle(frame, (x + S(4), y + h - S(4) - fh),
                  (x + w - S(4), y + h - S(4)), fill_bgr, -1, LINE_DRAW)


def draw_rpm_bar(frame, rpm, x, y, w, h, rpm_max=8000, red_start=7000):
    rect_fill(frame, x, y, x + w, y + h, (12, 12, 12))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    inner_x = x + S(8)
    inner_y = y + S(8)
    inner_w = w - S(16)
    inner_h = h - S(16)
    rs = int(inner_x + inner_w * (red_start / rpm_max))
    rect_fill(frame, inner_x, inner_y, rs, inner_y + inner_h, (40, 40, 40))
    rect_fill(frame, rs, inner_y, inner_x + inner_w,
              inner_y + inner_h, (0, 0, 255))
    rpm = float(np.clip(rpm, 0, rpm_max))
    px = int(inner_x + inner_w * (rpm / rpm_max))
    rect_fill(frame, inner_x, inner_y, px, inner_y + inner_h, (235, 235, 235))
    for r in range(0, rpm_max + 1, 500):
        tx = int(inner_x + inner_w * (r / rpm_max))
        major = (r % 1000 == 0)
        tick_h = inner_h if major else int(inner_h * 0.55)
        col = (0, 0, 0) if tx < px else (255, 255, 255)
        cv2.line(frame, (tx, inner_y + inner_h), (tx, inner_y + inner_h - tick_h), col, TH(2 if major else 1),
                 LINE_DRAW)
        if major and r > 0:
            label = str(r // 1000)
            (tw, _), _ = text_size(label, FONT, FS(0.5), TH(1))
            put_text(frame, label, (tx - tw // 2, y - S(6)), FONT,
                     FS(0.5), (255, 255, 255), TH(1), LINE_DRAW)


def draw_center(frame, x, y, w, h, rpm_txt, mph_txt, gear_txt, ign_txt, knock_txt):
    alpha_blend(frame, panel_rgba(w, h, alpha=110, shade=0), x, y)
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)

    put_text(frame, rpm_txt, (x + S(12), y + S(34)), FONT,
             FS(0.95), (255, 255, 255), TH(2), LINE_DRAW)
    put_text(frame, "RPM", (x + S(12), y + S(54)), FONT,
             FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)

    (mw, _), _ = text_size(mph_txt, FONT, FS(0.95), TH(2))
    put_text(frame, mph_txt, (x + w - S(12) - mw, y + S(34)),
             FONT, FS(0.95), (255, 255, 255), TH(2), LINE_DRAW)
    (lw, _), _ = text_size("MPH", FONT, FS(0.45), TH(1))
    put_text(frame, "MPH", (x + w - S(12) - lw, y + S(54)),
             FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)

    (tw, th), _ = text_size(gear_txt, FONT, FS(2.4), TH(4))
    cx = x + w // 2
    cy = y + h // 2
    put_text(frame, gear_txt, (cx - tw // 2, cy + th // 2 + S(10)), FONT, FS(2.4), (255, 255, 255), TH(4),
             LINE_DRAW)

    (gw, _), _ = text_size("GEAR", FONT, FS(0.55), TH(2))
    put_text(frame, "GEAR", (cx - gw // 2, cy + th // 2 + S(34)), FONT, FS(0.55), (210, 210, 210), TH(2),
             LINE_DRAW)

    put_text(frame, ign_txt, (x + S(12), y + h - S(28)), FONT,
             FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)
    put_text(frame, knock_txt, (x + S(12), y + h - S(10)),
             FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)


def draw_bottom_right(frame, data, x, y, w, h):
    alpha_blend(frame, panel_rgba(w, h, alpha=110, shade=0), x, y)
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    pad = S(10)
    left_w = S(150)
    right_w = S(150)
    tile_h = S(54)
    gap = S(10)
    stack_h = 4 * tile_h + 3 * gap
    base_center_w = w - left_w - right_w - 2 * pad
    center_w = int(base_center_w * 0.80)
    total_used = left_w + pad + center_w + pad + right_w
    start_x = x + (w - total_used) // 2
    stack_y = y + (h - stack_h) // 2
    lx = start_x
    cx0 = lx + left_w + pad
    rx = cx0 + center_w + pad

    draw_center(frame, cx0, stack_y, center_w, stack_h, data["rpm"], data["mph"], data["gear"], data["ign"],
                data["knock"])
    draw_tile(frame, lx, stack_y + 0 * (tile_h + gap),
              left_w, tile_h, "CLT", data["clt"])
    draw_tile(frame, lx, stack_y + 1 * (tile_h + gap),
              left_w, tile_h, "OIL", data["oil"])
    draw_tile(frame, lx, stack_y + 2 * (tile_h + gap),
              left_w, tile_h, "IAT", data["iat"])
    draw_tile(frame, lx, stack_y + 3 * (tile_h + gap),
              left_w, tile_h, "MAP", data["map"])
    draw_tile(frame, rx, stack_y + 0 * (tile_h + gap),
              right_w, tile_h, "TPS", data["tps"])
    draw_tile_fuel(frame, rx, stack_y + 1 * (tile_h + gap), right_w, tile_h, "Fuel Pres", data["fuel_psi"],
                   data["fuel_pct"])
    draw_tile(frame, rx, stack_y + 2 * (tile_h + gap), right_w,
              tile_h, "WB1/WB2", data["wb"], value_scale=0.62)
    draw_tile(frame, rx, stack_y + 3 * (tile_h + gap), right_w,
              tile_h, "Lambda Target", data["lt"], value_scale=0.62)
