import re
import time
import math
import numpy as np
import pandas as pd
import cv2

# ============================================================
# USER CONFIG (things you actually change)
# ============================================================

# --- Input / Output ---
NUMS = ["GX010055"]

NUM = NUMS[0]  # base name for your per-session files (e.g. 2.mp4 + 2.csv)

VIDEO_IN = f"{NUM}.mp4"  # input video
CSV_IN = f"{NUM}.csv"  # main telemetry CSV used by the overlay
LAP_CSV_IN = "session_20251220_093244_grange_v3.csv"  # RaceChrono export that contains GPS + lap info
VIDEO_OUT = f"{NUM}Final.mp4"  # output video

# --- Quick test mode ---
TEST_RENDER_FIRST_10S = False  # True: render only first TEST_DURATION_S seconds
TEST_DURATION_S = 120

# --- Audio trim nudge (seconds) ---
# If audio is still a hair late/early after the fix, tweak this:
#   +0.050 = delay audio 50ms
#   -0.050 = advance audio 50ms
AUDIO_TRIM_NUDGE_S = 0.0

# --- Encoder (scrub-friendly) ---
# 1.0 = force a keyframe about every second (fast scrubbing).
# Smaller = more keyframes (bigger file). Larger = fewer keyframes (smaller file but slower scrubbing).
SCRUB_KEYFRAME_EVERY_S = 1.0

# --- Optional small-file output mode ---
# Keeps the detected input resolution unchanged while using the same capped-
# bitrate H.264/AAC compression profile used for the standalone compressed clips.
COMPRESS_OUTPUT = True
COMPRESS_VIDEO_BITRATE = "7M"
COMPRESS_VIDEO_BUFFER_SIZE = "14M"
COMPRESS_AUDIO_BITRATE = "128k"
COMPRESS_NVENC_PRESET = "p6"
COMPRESS_X264_PRESET = "medium"

# --- Minimap ---

# --- Minimap dot smoothing (pixel-space) ---
MINIMAP_DOT_MAX_PX_STEP = 8.0  # max dot movement per frame in minimap pixels (prevents teleports)
MINIMAP_DOT_MIN_PX_STEP = 0.0  # set to ~0.8 if you want to avoid subpixel 'stalls'
MINIMAP_DOT_SMOOTH_ALPHA = 0.25
MINIMAP_DOT_STALL_EPS_PX = 0.15

MINIMAP_ENABLE = True
MINIMAP_POS = "topleft"  # topleft / topright / bottomleft / bottomright

# Minimap dot stability (prevents rare "teleport" snaps at hairpins / S-F)
# Smaller SEARCH_WINDOW_SEGS reduces the chance of snapping to a nearby parallel segment.
MINIMAP_DOT_SEARCH_WINDOW_SEGS = 90  # segments to search around last known segment (circular)
MINIMAP_DOT_MAX_JUMP_M = 25.0  # hard gate: max meters the dot may move in one frame
MINIMAP_DOT_MAX_STEP_PROG = 0.06  # gate on along-track progress step (0..1), cyclic (higher = more permissive)
MINIMAP_MAX_LATERAL_JUMP_M = 1.5  # hard cap on cross-track jump (m) to prevent parallel-segment flips
MINIMAP_CONFIRM_STREAK = 6  # frames required to accept a large lateral move
MINIMAP_CONFIRM_LATERAL_M = 1.5  # lateral threshold (m) for debounce
MINIMAP_CONFIRM_JUMP_M = 6.0  # total jump threshold (m) for debounce

# --- Virtual splits (recommended) ---
VIRTUAL_SPLITS_ENABLE = True
VIRTUAL_SPLITS_N = 6  # 3 / 4 / 6 / 8 etc.
VIRTUAL_SPLITS_RUNNING = True  # True: show a running delta inside the current sector
# Running-split sanity clamp: if delta goes insane early in lap, it's almost always a progress wrap/glitch.
SPLIT_SANITY_MAX_ABS_S = 10.0   # if |split| exceeds this, rebase progress for running split
SPLIT_SANITY_MAX_LAP_T_S = 25.0 # only apply rebasing within first N seconds of a lap

# --- Tire temperature color configuration (deg F) ---
# Blue -> Green (optimal) -> Yellow (warm) -> Red (hot)
# FRONT tires (TPMS valve-stem based)
FRONT_TIRE_TEMP_COLD_F = 110.0
FRONT_TIRE_TEMP_GREEN_LOW_F = 120.0
FRONT_TIRE_TEMP_GREEN_HIGH_F = 135.0
FRONT_TIRE_TEMP_YELLOW_LOW_F = 145.0
FRONT_TIRE_TEMP_YELLOW_HIGH_F = 160.0
FRONT_TIRE_TEMP_HOT_F = 175.0
# REAR tires
REAR_TIRE_TEMP_COLD_F = 105.0
REAR_TIRE_TEMP_GREEN_LOW_F = 115.0
REAR_TIRE_TEMP_GREEN_HIGH_F = 130.0
REAR_TIRE_TEMP_YELLOW_LOW_F = 140.0
REAR_TIRE_TEMP_YELLOW_HIGH_F = 155.0
REAR_TIRE_TEMP_HOT_F = 170.0

# --- Master UI scale ---
UI_SCALE = 1.5  # 1.0 = original size

DEBUG_PRINTS = False

# ============================================================

# --- Speed knobs (no visible quality loss for most UI) ---
FAST_NO_AA = True  # True: use LINE_8 for shapes (faster). Text is still anti-aliased.
LINE_DRAW = cv2.LINE_8 if FAST_NO_AA else cv2.LINE_AA

# -------------------- FAST TEXT CACHE --------------------
# cv2.putText is expensive per-frame. We cache rendered glyph bitmaps and blit them.
_TEXT_CACHE = {}

# allowed filenames

results = {}

with open("dataMarks.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) != 6:
            raise ValueError(f"Invalid format: {line}")

        file_name, trim_start, trim_end, video_sync_mmss, log_sync_mmss, fine_tune_ms = parts

        if file_name not in NUMS:
            continue

        results[file_name] = {
            "VIDEO_TRIM_START_S": trim_start,
            "VIDEO_TRIM_END_S": trim_end,
            "VIDEO_SYNC_AT_MMSS": video_sync_mmss,
            "LOG_SYNC_AT_MMSS": log_sync_mmss,
            "SYNC_FINE_TUNE_MS": int(fine_tune_ms),
        }

# --- Trim (SOURCE video time) ---
# Leave VIDEO_TRIM_END_MMSS=None to render to end of the source file.
VIDEO_TRIM_START_MMSS = results[NUM]['VIDEO_TRIM_START_S']  # mm:ss
VIDEO_TRIM_END_MMSS = results[NUM]['VIDEO_TRIM_END_S']  # mm:ss or None


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
        cv2.putText(patch, s, (x0, y0), fontFace, fontScale, (0, 0, 0), outline_th, lineType, bottomLeftOrigin)
        cv2.putText(patch, s, (x0, y0), fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

        alpha = (patch[:, :, 0] | patch[:, :, 1] | patch[:, :, 2]).astype(np.uint8)
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


# --- Derived times from USER CONFIG ---

def _mmss_to_s(mmss):
    if mmss is None:
        return None
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def _mmss_or_ss_to_s(v):
    """Parse 'm:ss', 'mm:ss', optionally with .mmm, or plain seconds string '20.4'."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    if re.match(r"^\d+(?:\.\d+)?$", s):
        return float(s)
    m = re.match(r"^(?P<m>\d+):(?P<s>\d{1,2})(?:\.(?P<ms>\d{1,3}))?$", s)
    if not m:
        raise ValueError(f"Bad time format: {v!r} (use seconds like 20.4 or m:ss(.mmm) like 0:20.400)")
    mm = int(m.group("m"))
    ss = int(m.group("s"))
    ms = m.group("ms")
    frac = 0.0
    if ms is not None:
        frac = int(ms.ljust(3, "0")) / 1000.0
    return mm * 60.0 + ss + frac


def _fmt_mmss_mmm(s: float) -> str:
    """Format seconds as m:ss.mmm."""
    if s is None:
        return "None"
    sign = "-" if s < 0 else ""
    s = abs(float(s))
    m = int(s // 60)
    sec = s - m * 60
    return f"{sign}{m}:{sec:06.3f}"


VIDEO_TRIM_START_S = _mmss_to_s(VIDEO_TRIM_START_MMSS)
VIDEO_TRIM_END_S = _mmss_to_s(VIDEO_TRIM_END_MMSS)

# -------------------- Manual anchor sync -> MAIN_START_S --------------------
VIDEO_SYNC_AT_MMSS = results[NUM]["VIDEO_SYNC_AT_MMSS"]
LOG_SYNC_AT_MMSS = results[NUM]["LOG_SYNC_AT_MMSS"]
SYNC_FINE_TUNE_MS = results[NUM]["SYNC_FINE_TUNE_MS"]

_video_anchor_s = _mmss_or_ss_to_s(VIDEO_SYNC_AT_MMSS)
_log_anchor_s = _mmss_or_ss_to_s(LOG_SYNC_AT_MMSS)

MAIN_START_S = float(_log_anchor_s) - float(_video_anchor_s) - (float(SYNC_FINE_TUNE_MS) / 1000.0)
print(
    f"[SYNC] video anchor = {VIDEO_SYNC_AT_MMSS} | "
    f"log anchor = {LOG_SYNC_AT_MMSS} | "
    f"fine tune = {SYNC_FINE_TUNE_MS:+d} ms | "
    f"MAIN_START = {_fmt_mmss_mmm(MAIN_START_S)}"
)


def _fmt_mmss_mmm(s):
    m = int(s // 60)
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


# -------------------- Performance --------------------
FAST_RENDER = True  # enable per-frame caching for speed (no quality drop)
TEXT_UPDATE_HZ = 10.0  # update expensive text strings at this rate (Hz)

# Units
KPA_TO_PSI = 0.1450377377
KPH_TO_MPH = 0.621371
ATM_PSI = 14.6959


def kpa_to_psi(v): return v * KPA_TO_PSI


def psi_abs_to_psig(psi_abs): return psi_abs - ATM_PSI


def kelvin_to_f(k): return (k - 273.15) * 9.0 / 5.0 + 32.0


FONT = cv2.FONT_HERSHEY_SIMPLEX


def S(px): return int(round(px * UI_SCALE))


def FS(f): return float(f * UI_SCALE)


def TH(t): return max(1, int(round(t * UI_SCALE)))


# -------------------- Text size cache (speed) --------------------
_TEXTSZ = {}


def text_size(txt, font, scale, thickness):
    k = (txt, float(scale), int(thickness))
    v = _TEXTSZ.get(k)
    if v is None:
        v = cv2.getTextSize(txt, font, scale, thickness)
        _TEXTSZ[k] = v
    return v


def time_to_seconds(t):
    if pd.isna(t):
        return np.nan
    s = str(t).strip()
    m = re.match(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<sec>\d{2})\.(?P<ms>\d+)$", s)
    if not m:
        return np.nan
    ms = int(m.group("ms")[:3].ljust(3, "0"))
    return int(m.group("h")) * 3600 + int(m.group("m")) * 60 + int(m.group("sec")) + ms / 1000.0


def pick(columns, pattern):
    for col in columns:
        if re.search(pattern, col, re.I):
            return col
    return None


def series_num(df, col):
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def detect_scale(df, col, kind):
    if col is None:
        return 1.0
    mx = float(np.nanmax(series_num(df, col)))
    if not np.isfinite(mx):
        return 1.0

    if kind == "ign":
        return 10.0 if mx > 100.0 else 1.0

    if kind == "lambda":
        if mx > 50 and mx <= 5000:
            return 1000.0
        if mx > 5 and mx <= 50:
            return 10.0
        if mx > 5000 and mx <= 50000:
            return 10000.0
        return 1.0

    if mx >= 1000 and mx < 10000:
        return 10.0
    if mx >= 10000:
        return 100.0
    return 1.0


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


_PANEL_CACHE = {}


# -------------------- ROI panel rendering (speed) --------------------
def draw_panel_roi(frame, x, y, w, h, draw_fn, *args, **kwargs):
    """Draw into a ROI VIEW (no copy). Fastest path: all drawing stays inside the smaller slice."""
    H, W = frame.shape[:2]
    x0 = max(0, int(x));
    y0 = max(0, int(y))
    x1 = min(W, x0 + int(w));
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
    put_text(frame, label, (x + S(12), y + S(20)), FONT, FS(0.40), (210, 210, 210), TH(1), LINE_DRAW)
    put_text(frame, value, (x + S(12), y + h - S(10)), FONT, FS(value_scale), (255, 255, 255), TH(2), LINE_DRAW)


def draw_tile_fuel(frame, x, y, w, h, label, psi_text, pct_text):
    rect_fill(frame, x, y, x + w, y + h, (18, 18, 18))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    put_text(frame, label, (x + S(12), y + S(20)), FONT, FS(0.40), (210, 210, 210), TH(1), LINE_DRAW)
    if pct_text:
        (tw, _), _ = text_size(pct_text, FONT, FS(0.35), TH(1))
        put_text(frame, pct_text, (x + w - S(12) - tw, y + S(20)), FONT, FS(0.35), (210, 210, 210), TH(1),
                 LINE_DRAW)
    put_text(frame, psi_text, (x + S(12), y + h - S(10)), FONT, FS(0.66), (255, 255, 255), TH(2), LINE_DRAW)


def draw_bar(frame, pct, x, y, w, h, fill_bgr):
    rect_fill(frame, x, y, x + w, y + h, (12, 12, 12))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    v = float(np.clip(pct, 0, 100))
    fh = int((h - S(8)) * (v / 100.0))
    cv2.rectangle(frame, (x + S(4), y + h - S(4) - fh), (x + w - S(4), y + h - S(4)), fill_bgr, -1, LINE_DRAW)


def draw_rpm_bar(frame, rpm, x, y, w, h, rpm_max=8000, red_start=7000):
    rect_fill(frame, x, y, x + w, y + h, (12, 12, 12))
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    inner_x = x + S(8);
    inner_y = y + S(8)
    inner_w = w - S(16);
    inner_h = h - S(16)
    rs = int(inner_x + inner_w * (red_start / rpm_max))
    rect_fill(frame, inner_x, inner_y, rs, inner_y + inner_h, (40, 40, 40))
    rect_fill(frame, rs, inner_y, inner_x + inner_w, inner_y + inner_h, (0, 0, 255))
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
            put_text(frame, label, (tx - tw // 2, y - S(6)), FONT, FS(0.5), (255, 255, 255), TH(1), LINE_DRAW)


def draw_center(frame, x, y, w, h, rpm_txt, mph_txt, gear_txt, ign_txt, knock_txt):
    alpha_blend(frame, panel_rgba(w, h, alpha=110, shade=0), x, y)
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)

    put_text(frame, rpm_txt, (x + S(12), y + S(34)), FONT, FS(0.95), (255, 255, 255), TH(2), LINE_DRAW)
    put_text(frame, "RPM", (x + S(12), y + S(54)), FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)

    (mw, _), _ = text_size(mph_txt, FONT, FS(0.95), TH(2))
    put_text(frame, mph_txt, (x + w - S(12) - mw, y + S(34)), FONT, FS(0.95), (255, 255, 255), TH(2), LINE_DRAW)
    (lw, _), _ = text_size("MPH", FONT, FS(0.45), TH(1))
    put_text(frame, "MPH", (x + w - S(12) - lw, y + S(54)), FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)

    (tw, th), _ = text_size(gear_txt, FONT, FS(2.4), TH(4))
    cx = x + w // 2;
    cy = y + h // 2
    put_text(frame, gear_txt, (cx - tw // 2, cy + th // 2 + S(10)), FONT, FS(2.4), (255, 255, 255), TH(4),
             LINE_DRAW)

    (gw, _), _ = text_size("GEAR", FONT, FS(0.55), TH(2))
    put_text(frame, "GEAR", (cx - gw // 2, cy + th // 2 + S(34)), FONT, FS(0.55), (210, 210, 210), TH(2),
             LINE_DRAW)

    put_text(frame, ign_txt, (x + S(12), y + h - S(28)), FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)
    put_text(frame, knock_txt, (x + S(12), y + h - S(10)), FONT, FS(0.45), (210, 210, 210), TH(1), LINE_DRAW)


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
    inset = int(0.20 * min(w, h))  # w/h already scaled; don't scale inset again
    rect_fill(frame, x1 + inset, y1 + inset, x2 - inset, y2 - inset, (25, 25, 25))
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
        t_raw = vals.get(tk, np.nan);
        p_raw = vals.get(pk, np.nan)
        t_k = (t_raw / scale_temp) if np.isfinite(t_raw) else np.nan
        t_f = kelvin_to_f(t_k) if np.isfinite(t_k) else np.nan
        color = tire_temp_to_color(t_f, axle=('rear' if tk in ('t_rl', 't_rr') else 'front'))
        p_kpa_abs = (p_raw / scale_tire_p) if np.isfinite(p_raw) else np.nan
        p_psi_abs = kpa_to_psi(p_kpa_abs) if np.isfinite(p_kpa_abs) else np.nan
        p_psig = psi_abs_to_psig(p_psi_abs) if np.isfinite(p_psi_abs) else np.nan
        if np.isfinite(p_psig): p_psig = max(0.0, p_psig)
        psi_text = f"{p_psig:0.1f}" if np.isfinite(p_psig) else "--"
        draw_tire_icon(frame, px + tw / 2, py + th / 2, tw, th, color, psi_text)


def draw_lap_panel(frame, lap_time_s, prev_lap_s, sess_best_s, split_s, lat_g, lon_g, txt_cache=None):
    # Bottom-left: times panel + a separate combined G widget to the right.
    # Only these two boxes get an extra scale bump.
    PANEL_SCALE = 1.20

    def S2(px):  # px in "base" pixels
        return int(round(px * UI_SCALE * PANEL_SCALE))

    def FS2(f):
        return float(f * UI_SCALE * PANEL_SCALE)

    def TH2(t):
        return max(1, int(round(t * UI_SCALE * PANEL_SCALE)))

    pad = S2(16)
    x = pad
    y = H - pad

    # ----- Times panel -----
    line_h = S2(26)
    w_time = S2(360)
    h_time = line_h * 6 + S2(18)  # Lap, Split (running), Prev (previous lap), Sess best, Day best
    y0 = y - h_time

    alpha_blend(frame, panel_rgba(w_time, h_time, alpha=110, shade=0), x, y0)
    rect_outline(frame, x, y0, x + w_time, y0 + h_time, (255, 255, 255), 1)

    lap_txt = f"Lap  {_fmt_laptime(lap_time_s)}"

    # Split line: keep the "Split ..." label color unchanged; only color the delta value.
    # (Also use darker colors that stay readable on the translucent panel.)
    split_label = "Split"
    split_val = "--.--"
    split_val_color = (200, 200, 200)
    if np.isfinite(split_s):
        split_sign = "-" if split_s < 0 else "+" if split_s > 0 else " "
        split_val = f"{split_sign}{abs(split_s):0.2f}"
        if split_s < 0:
            split_val_color = (0, 170, 0)  # darker green
        elif split_s > 0:
            split_val_color = (0, 0, 170)  # darker red/blue channel (BGR)

    # Prev-lap delta vs session-best (at the time). Blank until the first full session lap is completed.
    last_label = "Prev"
    last_val = "--.--"
    last_val_color = (200, 200, 200)
    try:
        if np.isfinite(_prev_lap_delta_s):
            ld = float(_prev_lap_delta_s)
            ld_sign = "-" if ld < 0 else "+" if ld > 0 else " "
            last_val = f"{ld_sign}{abs(ld):0.2f}"
            if ld < 0:
                last_val_color = (0, 170, 0)
            elif ld > 0:
                last_val_color = (0, 0, 170)
    except Exception:
        pass

    prev_txt = f"Prev {_fmt_laptime(prev_lap_s)}"
    sess_txt = f"Sess best {_fmt_laptime(sess_best_s)}"
    day_txt = f"Day best  {_fmt_laptime(DAY_BEST_LAP_S)}"

    put_text(frame, lap_txt, (x + S2(12), y0 + S2(30)), FONT, FS2(0.62), (255, 255, 255), TH2(2), LINE_DRAW)

    # Draw "Split" label (neutral), then delta value (colored)
    split_y = y0 + S2(30) + 1 * line_h
    label_pos = (x + S2(12), split_y)
    put_text(frame, split_label, label_pos, FONT, FS2(0.50), (210, 210, 210), TH2(2), LINE_DRAW)

    # Compute an x-offset so the value starts after the label
    try:
        (tw, th), _ = cv2.getTextSize(split_label + " ", FONT, FS2(0.50), TH2(2))
    except Exception:
        tw = S2(70)
    val_pos = (label_pos[0] + int(tw), split_y)
    put_text(frame, split_val, val_pos, FONT, FS2(0.50), split_val_color, TH2(2), LINE_DRAW)

    put_text(frame, prev_txt, (x + S2(12), y0 + S2(30) + 2 * line_h), FONT, FS2(0.46), (210, 210, 210), TH2(1),
             LINE_DRAW)

    # Draw "Prev" label (neutral), then value (colored)
    last_y = y0 + S2(30) + 3 * line_h
    put_text(frame, last_label, (x + S2(12), last_y), FONT, FS2(0.46), (210, 210, 210), TH2(1), LINE_DRAW)
    try:
        (tw2, th2), _ = cv2.getTextSize(last_label + " ", FONT, FS2(0.46), TH2(1))
    except Exception:
        tw2 = S2(70)
    put_text(frame, last_val, (x + S2(12) + int(tw2), last_y), FONT, FS2(0.46), last_val_color, TH2(1), LINE_DRAW)
    put_text(frame, sess_txt, (x + S2(12), y0 + S2(30) + 4 * line_h), FONT, FS2(0.46), (210, 210, 210), TH2(1),
             LINE_DRAW)
    put_text(frame, day_txt, (x + S2(12), y0 + S2(30) + 5 * line_h), FONT, FS2(0.46), (210, 210, 210), TH2(1),
             LINE_DRAW)

    # ----- G widget panel (to the right of times) -----
    gap = S2(10)
    w_g = S2(240)
    h_g = h_time
    xg = x + w_time + gap
    yg0 = y - h_g

    alpha_blend(frame, panel_rgba(w_g, h_g, alpha=110, shade=0), xg, yg0)
    rect_outline(frame, xg, yg0, xg + w_g, yg0 + h_g, (255, 255, 255), 1)

    # 2D G ring: x = Lat, y = Lon (positive Lon = accel; negative = brake)
    lat = float(lat_g) if np.isfinite(lat_g) else 0.0
    lon = float(lon_g) if np.isfinite(lon_g) else 0.0

    # Display tuning (visual saturation)
    G_RING_MAX = 1.80

    # Ring geometry
    inner_pad = S2(18)
    # Text occupies left side; place the circle cluster more to the right to avoid overlap.
    text_block_w = S2(130)
    cx = xg + int(round(w_g * 0.72))
    cy = yg0 + h_g // 2 + S2(6)

    # Base radius from panel size, then clamp so left edge clears the text block.
    r_base = int(max(1, min(w_g, h_g) // 2 - inner_pad))
    r_max_left = max(1, cx - (xg + text_block_w))
    r_max_right = max(1, (xg + w_g - inner_pad) - cx)
    r = int(max(1, min(r_base, r_max_left, r_max_right)))

    # Rings + axes
    cv2.circle(frame, (cx, cy), r, (210, 210, 210), TH2(2), LINE_DRAW)
    cv2.circle(frame, (cx, cy), int(r * 0.5), (120, 120, 120), TH2(1), LINE_DRAW)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), (120, 120, 120), TH2(1), LINE_DRAW)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), (120, 120, 120), TH2(1), LINE_DRAW)

    # Vector + dot
    gx = float(np.clip(lat, -G_RING_MAX, G_RING_MAX))
    gy = float(np.clip(lon, -G_RING_MAX, G_RING_MAX))
    bx = int(round(cx + (gx / G_RING_MAX) * r))
    by = int(round(cy - (gy / G_RING_MAX) * r))

    cv2.line(frame, (cx, cy), (bx, by), (255, 255, 255), TH2(2), LINE_DRAW)
    dot_r = S2(10)
    cv2.circle(frame, (bx, by), dot_r, (255, 255, 255), -1, LINE_DRAW)
    cv2.circle(frame, (bx, by), dot_r, (0, 0, 0), TH2(2), LINE_DRAW)

    # Magnitude ring arc (combined vector)
    mag = float(np.hypot(lat, lon)) if (np.isfinite(lat_g) and np.isfinite(lon_g)) else float("nan")
    mag_norm = float(np.clip((mag / G_RING_MAX) if np.isfinite(mag) else 0.0, 0.0, 1.0))
    # Arc from top (-90 deg) clockwise
    start_ang = -90
    end_ang = int(round(start_ang + 360 * mag_norm))
    cv2.ellipse(frame, (cx, cy), (r, r), 0, start_ang, end_ang, (255, 255, 255), TH2(2), LINE_DRAW)

    # Text layout: main |G| top-left (bold bigger), then lat, then lon
    mag_str = f"{mag:0.2f}g" if np.isfinite(mag) else "--.--g"
    lat_str = f"Lat {lat:+0.2f}g" if np.isfinite(lat_g) else "Lat --.--g"
    lon_str = f"Lon {lon:+0.2f}g" if np.isfinite(lon_g) else "Lon --.--g"

    put_text(frame, mag_str, (xg + S2(12), yg0 + S2(30)), FONT, FS2(0.62), (255, 255, 255), TH2(2), LINE_DRAW)
    put_text(frame, lat_str, (xg + S2(12), yg0 + S2(30) + 1 * line_h), FONT, FS2(0.46), (210, 210, 210), TH2(1),
             LINE_DRAW)
    put_text(frame, lon_str, (xg + S2(12), yg0 + S2(30) + 2 * line_h), FONT, FS2(0.46), (210, 210, 210), TH2(1),
             LINE_DRAW)


def draw_bottom_right(frame, data, x, y, w, h):
    alpha_blend(frame, panel_rgba(w, h, alpha=110, shade=0), x, y)
    rect_outline(frame, x, y, x + w, y + h, (255, 255, 255), 1)
    pad = S(10);
    left_w = S(150);
    right_w = S(150)
    tile_h = S(54);
    gap = S(10)
    stack_h = 4 * tile_h + 3 * gap
    base_center_w = w - left_w - right_w - 2 * pad
    center_w = int(base_center_w * 0.80)
    total_used = left_w + pad + center_w + pad + right_w
    start_x = x + (w - total_used) // 2
    stack_y = y + (h - stack_h) // 2
    lx = start_x;
    cx0 = lx + left_w + pad;
    rx = cx0 + center_w + pad

    draw_center(frame, cx0, stack_y, center_w, stack_h, data["rpm"], data["mph"], data["gear"], data["ign"],
                data["knock"])
    draw_tile(frame, lx, stack_y + 0 * (tile_h + gap), left_w, tile_h, "CLT", data["clt"])
    draw_tile(frame, lx, stack_y + 1 * (tile_h + gap), left_w, tile_h, "OIL", data["oil"])
    draw_tile(frame, lx, stack_y + 2 * (tile_h + gap), left_w, tile_h, "IAT", data["iat"])
    draw_tile(frame, lx, stack_y + 3 * (tile_h + gap), left_w, tile_h, "MAP", data["map"])
    draw_tile(frame, rx, stack_y + 0 * (tile_h + gap), right_w, tile_h, "TPS", data["tps"])
    draw_tile_fuel(frame, rx, stack_y + 1 * (tile_h + gap), right_w, tile_h, "Fuel Pres", data["fuel_psi"],
                   data["fuel_pct"])
    draw_tile(frame, rx, stack_y + 2 * (tile_h + gap), right_w, tile_h, "WB1/WB2", data["wb"], value_scale=0.62)
    draw_tile(frame, rx, stack_y + 3 * (tile_h + gap), right_w, tile_h, "Lambda Target", data["lt"], value_scale=0.62)


# -------------------- Load CSV --------------------
channels = []
data_start = None
with open(CSV_IN, "r", errors="ignore") as f:
    for idx, line in enumerate(f):
        if line.startswith("Channel :"):
            channels.append(line.split("Channel :", 1)[1].strip())
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3},", line):
            data_start = idx
            break

names = ["Log Time"] + channels
log_df = pd.read_csv(CSV_IN, skiprows=data_start, names=names, engine="python")

t_log = log_df["Log Time"].apply(time_to_seconds).to_numpy()
valid = np.isfinite(t_log)
log_df = log_df.loc[valid].copy()
t_log = t_log[valid]
order = np.argsort(t_log)
log_df = log_df.iloc[order].reset_index(drop=True)
t_log = t_log[order]
# Normalize log time so the first sample is t=0.0s (Haltech logs often start at non-zero HH:MM:SS)
_t0 = float(t_log[0])
t_log = t_log - _t0

LOG_START_S = float(np.nanmin(t_log))

COLS = log_df.columns
COL = {
    "rpm": pick(COLS, r"^RPM$"),
    "ign": pick(COLS, r"Ignition 1 Angle"),
    "knock": pick(COLS, r"Knock Sensor 1 Knock Count"),
    "tps": pick(COLS, r"Throttle Position"),
    "brake": pick(COLS, r"Brake"),
    "map": pick(COLS, r"Manifold Pressure"),
    "speed": pick(COLS, r"Vehicle Speed"),
    "gear": pick(COLS, r"^Gear$"),
    "wb1": pick(COLS, r"Wideband O2 1"),
    "wb2": pick(COLS, r"Wideband O2 2"),
    "lambda_tgt": pick(COLS, r"Target Lambda"),
    "iat": pick(COLS, r"Intake Air Temperature"),
    "clt": pick(COLS, r"Coolant Temperature$"),
    "oil_temp": pick(COLS, r"Oil Temperature"),
    "fuel_p": pick(COLS, r"^Fuel Pressure$"),
    "fuel_p_exp": pick(COLS, r"Fuel Pressure Expected"),
    "p_fl": pick(COLS, r"Front Left Tyre Pressure"),
    "p_fr": pick(COLS, r"Front Right Tyre Pressure"),
    "p_rl": pick(COLS, r"Rear Left Tyre Pressure"),
    "p_rr": pick(COLS, r"Rear Right Tyre Pressure"),
    "t_fl": pick(COLS, r"Front Left Tyre Temperature"),
    "t_fr": pick(COLS, r"Front Right Tyre Temperature"),
    "t_rl": pick(COLS, r"Rear Left Tyre Temperature"),
    "t_rr": pick(COLS, r"Rear Right Tyre Temperature"),
}

SCALE = {
    "speed": detect_scale(log_df, COL.get("speed"), "speed"),
    "kpa": detect_scale(log_df, COL.get("map"), "kpa"),
    "fuel_kpa": detect_scale(log_df, COL.get("fuel_p"), "kpa"),
    "fuel_kpa_exp": detect_scale(log_df, COL.get("fuel_p_exp"), "kpa"),
    "tire_kpa": max(
        detect_scale(log_df, COL.get("p_fl"), "kpa"),
        detect_scale(log_df, COL.get("p_fr"), "kpa"),
        detect_scale(log_df, COL.get("p_rl"), "kpa"),
        detect_scale(log_df, COL.get("p_rr"), "kpa"),
    ),
    "temp_k": max(
        detect_scale(log_df, COL.get("clt"), "temp_k"),
        detect_scale(log_df, COL.get("iat"), "temp_k"),
        detect_scale(log_df, COL.get("oil_temp"), "temp_k"),
        detect_scale(log_df, COL.get("t_fl"), "temp_k"),
        detect_scale(log_df, COL.get("t_fr"), "temp_k"),
        detect_scale(log_df, COL.get("t_rl"), "temp_k"),
        detect_scale(log_df, COL.get("t_rr"), "temp_k"),
    ),
    "lambda": detect_scale(log_df, COL.get("wb1"), "lambda"),
    "tps": detect_scale(log_df, COL.get("tps"), "pct"),
}

IGN_SCALE = 10.0 if (COL.get("ign") and float(np.nanmax(series_num(log_df, COL["ign"]))) > 100.0) else 1.0
KNOCK_SCALE = 10.0 if (COL.get("knock") and float(np.nanmax(series_num(log_df, COL["knock"]))) > 1000.0) else 1.0

# -------------------- Cache interpolated arrays once (speed) --------------------
CACHE_SERIES = {}
for key, colname in COL.items():
    if not colname:
        continue
    try:
        CACHE_SERIES[key] = series_num(log_df, colname).interpolate(limit_direction="both").to_numpy()
    except Exception:
        pass


def v_interp(key, t_sample):
    arr = CACHE_SERIES.get(key)
    if arr is None:
        return np.nan
    return float(np.interp(t_sample, t_log, arr))


# -------------------- Brake channel normalization --------------------
# Your Haltech "Brake" channel may be:
#   - 0..100 already (percent)
#   - 0..1 (fraction)
#   - a pressure sensor in kPa/bar/psi (large numbers)
# We normalize it to 0..100 for the vertical bar.
BRAKE_SCALE_MODE = "auto"  # "auto" or one of: "pct", "frac", "pressure"
BRAKE_PRESSURE_P99_PCT = 99.7  # percentile used as "100%" when brake is a pressure sensor
BRAKE_PRESSURE_HEADROOM = 1.40  # >1.0 gives headroom so the bar doesn't peg early
BRAKE_PRESSURE_MIN_CLAMP = 0.0

# -------------------- Fast per-frame channel cache --------------------
CHANNEL_CACHE = {}  # key -> np.ndarray sampled at each rendered frame
CURRENT_FRAME_I = -1  # set in the main loop


def v(key, t_sample):
    """Value lookup with optional per-frame caching (FAST_RENDER)."""
    if FAST_RENDER and CURRENT_FRAME_I >= 0:
        arr = CHANNEL_CACHE.get(key)
        if arr is not None and 0 <= CURRENT_FRAME_I < len(arr):
            return arr[CURRENT_FRAME_I]
    return v_interp(key, t_sample)


def _brake_pct_from_raw(raw: float) -> float:
    if not np.isfinite(raw):
        return 0.0
    x = float(raw)
    # Auto-detect based on distribution
    if BRAKE_SCALE_MODE == "pct":
        pct = x
    elif BRAKE_SCALE_MODE == "frac":
        pct = x * 100.0
    elif BRAKE_SCALE_MODE == "pressure":
        denom = float(_BRAKE_PRESSURE_P99) if np.isfinite(_BRAKE_PRESSURE_P99) and _BRAKE_PRESSURE_P99 > 1e-6 else 1.0
        denom *= float(BRAKE_PRESSURE_HEADROOM)
        pct = (max(BRAKE_PRESSURE_MIN_CLAMP, x) / denom) * 100.0
    else:
        # auto
        if _BRAKE_MAX <= 1.5:
            pct = x * 100.0
        elif _BRAKE_MAX <= 130.0:
            pct = x
        else:
            denom = float(_BRAKE_PRESSURE_P99) if np.isfinite(_BRAKE_PRESSURE_P99) and _BRAKE_PRESSURE_P99 > 1e-6 else (
                _BRAKE_MAX if _BRAKE_MAX > 1e-6 else 1.0)
            denom *= float(BRAKE_PRESSURE_HEADROOM)
            pct = (max(BRAKE_PRESSURE_MIN_CLAMP, x) / denom) * 100.0
    return float(np.clip(pct, 0.0, 100.0))


# Precompute brake stats once
_br = CACHE_SERIES.get("brake")
if _br is None:
    _BRAKE_MAX = 0.0
    _BRAKE_PRESSURE_P99 = 1.0
else:
    _br_f = _br[np.isfinite(_br)]
    if _br_f.size == 0:
        _BRAKE_MAX = 0.0
        _BRAKE_PRESSURE_P99 = 1.0
    else:
        _BRAKE_MAX = float(np.nanmax(_br_f))
        _BRAKE_PRESSURE_P99 = float(np.nanpercentile(_br_f, BRAKE_PRESSURE_P99_PCT))
        if not np.isfinite(_BRAKE_PRESSURE_P99) or _BRAKE_PRESSURE_P99 <= 1e-6:
            _BRAKE_PRESSURE_P99 = max(_BRAKE_MAX, 1.0)


def brake_pct_at(t_sample: float) -> float:
    return _brake_pct_from_raw(v("brake", t_sample))


def _speed_mph_at(t_sample: float) -> float:
    """Main log vehicle speed at t_sample, returned in MPH."""
    s_raw = v("speed", t_sample)
    if not np.isfinite(s_raw):
        return float("nan")
    s_kph = (s_raw / SCALE["speed"])
    return float(s_kph * KPH_TO_MPH)

    t_min = float(np.nanmin(t_log))
    t_max = float(np.nanmax(t_log))

    lo = max(t_min, t_guess - float(SPEED_SEARCH_WINDOW_S))
    hi = min(t_max, t_guess + float(SPEED_SEARCH_WINDOW_S))
    if hi <= lo + 1.0:
        return t_guess

    step = 0.10  # 10 Hz search
    grid = np.arange(lo, hi, step, dtype=np.float64)

    best_t = t_guess
    best_score = float("inf")

    for t0 in grid:
        v0 = _speed_mph_at(float(t0))
        if not np.isfinite(v0):
            continue

        # prefer near target (usually 0 mph at video start)
        score = abs(v0 - float(START_SPEED_TARGET_MPH))

        # prefer "rising" shortly after (leaving pits / launch)
        t1 = float(min(t0 + float(SPEED_RISE_LOOKAHEAD_S), t_max))
        v1 = _speed_mph_at(t1)
        if np.isfinite(v1):
            dv = v1 - v0
            if dv < float(SPEED_RISE_MIN_DELTA_MPH):
                score += 5.0  # heavy penalty if not rising
        else:
            score += 2.0

        # tiny preference for staying close to the original guess
        score += 0.01 * abs(float(t0) - float(t_guess))

        if score < best_score:
            best_score = score
            best_t = float(t0)

    return best_t

    # Guess where the event should be in main-log time.
    t_guess_event = float(t_main_start) + float(EVENT_VIDEO_S)

    t_min = float(np.nanmin(t_log))
    t_max = float(np.nanmax(t_log))

    lo = max(t_min, t_guess_event - float(EVENT_SEARCH_WINDOW_S))
    hi = min(t_max, t_guess_event + float(EVENT_SEARCH_WINDOW_S))
    if hi <= lo + 0.5:
        return 0.0

    step = 0.02  # 50 Hz scan for sharp transitions
    grid = np.arange(lo, hi, step, dtype=np.float64)

    best_t = None
    best_score = float("inf")

    for t0 in grid:
        tps0 = v("tps", float(t0))
        if not np.isfinite(tps0):
            continue
        tps0 = float(tps0 / SCALE["tps"])
        if tps0 > float(EVENT_TPS_LOW_MAX):
            continue  # not a "zero TPS" moment

        t_prev = float(max(t_min, t0 - float(EVENT_TPS_LOOKBACK_S)))
        tps_prev = v("tps", t_prev)
        if not np.isfinite(tps_prev):
            continue
        tps_prev = float(tps_prev / SCALE["tps"])

        # We want "was high -> now low"
        if tps_prev < float(EVENT_TPS_PREV_HIGH_MIN):
            continue

        # Score: prefer lower TPS now, higher TPS before, and closer to the guessed time.
        score = (tps0 * 2.0) - (tps_prev * 0.2) + (0.05 * abs(float(t0) - t_guess_event))
        if score < best_score:
            best_score = score
            best_t = float(t0)

    if best_t is None:
        print("Event sync: no suitable TPS-drop event found near EVENT_VIDEO_S; no adjustment applied.")
        return 0.0

    delta = best_t - t_guess_event
    print(f"Event sync: adjusted by {delta:+.3f}s (event at log t={best_t:.3f} -> video t={EVENT_VIDEO_S:.3f}s)")
    return float(delta)


# -------------------- Video IO --------------------
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_IN}. Put it in the same folder as this script.")

# -------------------- Robustly probe fps/size when OpenCV lies (match source fps/res) --------------------
import subprocess, shutil, os, json, math


def probe_video_props(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", path
    ]
    try:
        out = subprocess.check_output(cmd)
        j = json.loads(out)
        s = j["streams"][0]

        def parse_rate(r):
            if not r or r == "0/0" or r == "0/1":
                return 0.0
            num, den = r.split("/")
            den = float(den)
            return float(num) / den if den else 0.0

        return {
            "W": int(s.get("width", 0) or 0),
            "H": int(s.get("height", 0) or 0),
            "fps_avg": parse_rate(s.get("avg_frame_rate", "0/1")),
            "fps_r": parse_rate(s.get("r_frame_rate", "0/1")),
        }
    except Exception:
        return None


def start_ffmpeg_frame_reader(video_in: str, W: int, H: int, fps: float, start_s: float):
    """Decode frames with ffmpeg and stream raw BGR24 frames to Python (faster/more reliable than OpenCV on Windows)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH. Install it (e.g. winget install Gyan.FFmpeg).")

    # -ss before -i for fast seek; good enough for overlay work
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.6f}",
        "-i", video_in,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-vf", f"fps={fps:.9f}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
    frame_bytes = int(W) * int(H) * 3

    def read_frame():
        buf = p.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape((H, W, 3)).copy()

    return p, read_frame


fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

p = probe_video_props(VIDEO_IN)
if p:
    # Prefer probe dimensions if OpenCV returned 0
    if W <= 0 and p["W"] > 0: W = p["W"]
    if H <= 0 and p["H"] > 0: H = p["H"]

    # Prefer avg_frame_rate if OpenCV fps missing/bad
    if fps <= 1e-3 or not np.isfinite(fps):
        if p["fps_avg"] > 1e-3:
            fps = p["fps_avg"]
        elif p["fps_r"] > 1e-3:
            fps = p["fps_r"]

if fps <= 1e-3 or not np.isfinite(fps):
    fps = 60.0  # last resort fallback

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

max_frames = total_frames if total_frames > 0 else 10 ** 12
if TEST_RENDER_FIRST_10S:
    max_frames = min(max_frames, int(TEST_DURATION_S * fps))

# Apply trim: seek to VIDEO_TRIM_START_S and limit max_frames to VIDEO_TRIM_END_S
video_duration_s = (total_frames / fps) if (total_frames > 0 and fps > 1e-6) else None

trim_start_s = float(max(0.0, VIDEO_TRIM_START_S))
trim_end_s = None if VIDEO_TRIM_END_S is None else float(max(trim_start_s, VIDEO_TRIM_END_S))

# Print durations considering trims and (optionally) test-mode cut
try:
    _end_s = video_duration_s if trim_end_s is None else trim_end_s
    if _end_s is not None:
        _trimmed_dur_s = max(0.0, float(_end_s) - float(trim_start_s))
        _final_out_dur_s = _trimmed_dur_s
        if TEST_RENDER_FIRST_10S:
            _final_out_dur_s = min(_final_out_dur_s, float(TEST_DURATION_S))

        if TEST_RENDER_FIRST_10S and (_final_out_dur_s < _trimmed_dur_s - 1e-6):
            print(
                f"[TRIM] start={_fmt_mmss_mmm(trim_start_s)} end={_fmt_mmss_mmm(_end_s)} | "
                f"trimmed_dur={_fmt_mmss_mmm(_trimmed_dur_s)} ({_trimmed_dur_s:.3f}s) | "
                f"FINAL_out_dur={_fmt_mmss_mmm(_final_out_dur_s)} ({_final_out_dur_s:.3f}s) [TEST MODE]"
            )
        else:
            print(
                f"[TRIM] start={_fmt_mmss_mmm(trim_start_s)} end={_fmt_mmss_mmm(_end_s)} -> "
                f"output_dur={_fmt_mmss_mmm(_final_out_dur_s)} ({_final_out_dur_s:.3f}s)"
            )
    else:
        print(f"[TRIM] start={_fmt_mmss_mmm(trim_start_s)} end=None -> output_dur=unknown")
except Exception as _e:
    print(f"[TRIM] (could not compute output duration: {_e})")

try:
    _end_s = video_duration_s if trim_end_s is None else trim_end_s
    if _end_s is not None:
        _out_dur_s = max(0.0, float(_end_s) - float(trim_start_s))
        if TEST_RENDER_FIRST_10S:
            _out_dur_s = min(float(TEST_DURATION_S), _out_dur_s)
    else:
        pass
except Exception as _e:
    pass

# Seek OpenCV capture to the trimmed start so we don't decode unused leading video.
if trim_start_s > 0:
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, trim_start_s * 1000.0)
    except Exception:
        pass

# ffmpeg audio input trim to match VIDEO_TRIM_* (keeps A/V aligned)
AUDIO_TRIM_START_S = trim_start_s
AUDIO_TRIM_DUR_S = None if trim_end_s is None else max(0.0, trim_end_s - trim_start_s)

# Seek to start
if trim_start_s > 0.0:
    # OpenCV time seeking can be unreliable for MP4s; seek by frame first, then msec as fallback.
    start_frame = int(round(trim_start_s * fps))
    ok_seek = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not ok_seek:
        cap.set(cv2.CAP_PROP_POS_MSEC, trim_start_s * 1000.0)

# Compute frames available under trim
if trim_end_s is not None:
    max_frames = int(max(0.0, (trim_end_s - trim_start_s)) * fps)
else:
    if video_duration_s is not None:
        max_frames = int(max(0.0, (video_duration_s - trim_start_s)) * fps)

# Re-apply test duration after trim
if TEST_RENDER_FIRST_10S:
    max_frames = min(max_frames, int(TEST_DURATION_S * fps))

# -------------------- Lap/G-force (RaceChrono) --------------------
# We align the RaceChrono "day CSV" to the Haltech log by matching MPH shape (normalized cross-correlation).
# Then we sample lap_time + lateral/longitudinal g for the video window.
RACECHRONO_MPS_TO_MPH = 2.2369362920544


def _load_racechrono_csv(path):
    # RaceChrono has a small text header then a CSV header line starting with "timestamp,"
    header_idx = None
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lower().startswith("timestamp,"):
                header_idx = i
                break
    if header_idx is None:
        raise SystemExit(f"Could not find RaceChrono header row in: {path}")

    df = pd.read_csv(path, skiprows=header_idx, engine="python")
    # First 2 rows after header are units/source rows (non-numeric "timestamp")
    df["timestamp_num"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.loc[np.isfinite(df["timestamp_num"])].copy()

    # Pick speed column (RaceChrono sometimes has duplicates like speed & speed.1)
    speed_col = "speed.1" if "speed.1" in df.columns else ("speed" if "speed" in df.columns else None)
    if speed_col is None:
        raise SystemExit("RaceChrono CSV missing speed column.")

    def col_or_none(name):
        return name if name in df.columns else None

    out = {
        "t_unix": pd.to_numeric(df["timestamp_num"], errors="coerce").to_numpy(dtype=np.float64),
        "speed_mph": pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64) * RACECHRONO_MPS_TO_MPH,
        "lap_number": pd.to_numeric(df.get("lap_number", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "lat_g": pd.to_numeric(df.get("lateral_acc", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "lon_g": pd.to_numeric(df.get("longitudinal_acc", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        # GPS position (if present in export)
        "gps_lat": pd.to_numeric(df.get("latitude", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "gps_lon": pd.to_numeric(df.get("longitude", np.nan), errors="coerce").to_numpy(dtype=np.float64),
    }
    # Basic cleanup
    m = np.isfinite(out["t_unix"]) & np.isfinite(out["speed_mph"])
    for k in out:
        out[k] = out[k][m]
    return out


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


# Build lap sync mapping (t_sample in Haltech "seconds of day") -> RaceChrono unix timestamp
lap_data = _load_racechrono_csv(LAP_CSV_IN)
lap_t = lap_data["t_unix"]
lap_speed = lap_data["speed_mph"]
lap_lapnum = lap_data["lap_number"]
lap_lat = lap_data["lat_g"]
lap_lon = lap_data["lon_g"]
lap_gps_lat = lap_data.get("gps_lat", None)
lap_gps_lon = lap_data.get("gps_lon", None)

# -------------------- Minimap (GPS) --------------------
# Start/finish alignment tweak (use if split line doesn't match true start/finish)
MINIMAP_SF_SHIFT_M = -12.95  # +meters: move drawn start/finish line forward along travel; -meters moves it backward
MINIMAP_W_BASE = 240  # base pixels at UI_SCALE=1.0
MINIMAP_H_BASE = 240
MINIMAP_MARGIN_BASE = 18
MINIMAP_PAD_BASE = 10
MINIMAP_TRACK_THICKNESS_BASE = 2
MINIMAP_SF_THICKNESS_BASE = 1  # thinner start/finish marker
MINIMAP_SF_HALF_LEN_PX = 10  # half-length (px @ UI_SCALE=1) of the start/finish marker
MINIMAP_DOT_R_BASE = 4

# Track-line cleanup knobs
# Goal: draw ONE clean "centerline-ish" loop and ignore pit-lane / out-and-back / paddock lines.
MINIMAP_MIN_SPEED_MPH = 25.0  # ignore very slow GPS points (pit/grid/paddock)
MINIMAP_DBSCAN_EPS_M = 8.0  # clustering radius in meters (separates pit lane if it's offset enough)
MINIMAP_DBSCAN_MIN_SAMPLES = 60  # minimum points to form a cluster
MINIMAP_SIMPLIFY_EPS_M = 1.2  # RDP simplification tolerance in meters (smaller = more detailed line)
MINIMAP_SMOOTH_WINDOW = 5  # moving-average window (odd integer recommended)

# Shape control
MINIMAP_CORNER_SMOOTH_ITERS = 2  # Chaikin smoothing iterations (corners less sharp)
MINIMAP_JUMP_THRESH_M = 18.0  # remove big discontinuities/blips (meters)
MINIMAP_SNAP_DOT_TO_LINE = True  # forces red dot to sit on the drawn track line

# Pre-render static minimap background + track polyline (fast per-frame)
_minimap_base = None
_minimap_transform = None  # (lon0, lat0, cos_lat0, x_min, y_min, s, pad, w, h)
_minimap_centerline_m = None  # Nx2 meters polyline (same coordinate space as transform)
_minimap_centerline_px = None  # Nx2 pixels polyline


def _build_minimap_base():
    global _minimap_base, _minimap_transform, _minimap_centerline_m, _minimap_centerline_px
    if not MINIMAP_ENABLE:
        return
    if lap_gps_lat is None or lap_gps_lon is None:
        print("Minimap: GPS latitude/longitude not found in RaceChrono CSV -> disabled.")
        return

    lat_raw = np.asarray(lap_gps_lat, dtype=np.float64)
    lon_raw = np.asarray(lap_gps_lon, dtype=np.float64)
    spd_raw = np.asarray(lap_speed, dtype=np.float64)
    ln_raw = np.asarray(lap_lapnum, dtype=np.float64)

    m = np.isfinite(lat_raw) & np.isfinite(lon_raw) & np.isfinite(spd_raw)
    m &= (spd_raw >= float(MINIMAP_MIN_SPEED_MPH))
    m &= np.isfinite(ln_raw) & (ln_raw >= 1)

    if int(np.sum(m)) < 200:
        print("Minimap: not enough valid GPS samples after filtering -> disabled.")
        return

    lat = lat_raw[m]
    lon = lon_raw[m]
    ln = ln_raw[m].astype(np.int64, copy=False)

    # --- Project to local meters (tangent-plane-ish) ---
    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    cos_lat0 = float(np.cos(np.deg2rad(lat0)))
    M_PER_DEG = 111320.0
    x_m = (lon - lon0) * cos_lat0 * M_PER_DEG
    y_m = (lat - lat0) * M_PER_DEG

    # --- Cluster and keep main racing-surface cluster ---
    try:
        from sklearn.cluster import DBSCAN
        XY = np.column_stack([x_m, y_m])
        labels = DBSCAN(eps=float(MINIMAP_DBSCAN_EPS_M), min_samples=int(MINIMAP_DBSCAN_MIN_SAMPLES)).fit_predict(XY)
        lab, cnt = np.unique(labels[labels >= 0], return_counts=True)
        if len(lab) == 0:
            main_mask = np.ones_like(labels, dtype=bool)
        else:
            main_label = int(lab[int(np.argmax(cnt))])
            main_mask = labels == main_label
    except Exception:
        main_mask = np.ones_like(x_m, dtype=bool)

    if int(np.sum(main_mask)) < 200:
        print("Minimap: clustering removed too many points -> disabled.")
        return

    x_m = x_m[main_mask]
    y_m = y_m[main_mask]
    ln = ln[main_mask]

    # --- Choose one representative lap (most points among laps with normal duration) to avoid multi-lap spaghetti ---
    _valid_ln = np.isfinite(lap_lapnum) & (lap_lapnum >= 1)
    _ln_ints = np.round(lap_lapnum[_valid_ln]).astype(int)
    _ts = lap_t[_valid_ln]
    _durations = {l: float(_ts[_ln_ints == l][-1] - _ts[_ln_ints == l][0]) for l in np.unique(_ln_ints)}
    _valid_durs = [d for d in _durations.values() if 20.0 < d < 600.0]
    _med_dur = float(np.median(_valid_durs)) if _valid_durs else 100.0

    uniq_ln, ln_cnt = np.unique(ln, return_counts=True)
    valid_cands = [i for i, l in enumerate(uniq_ln) if l in _durations and 0.5 * _med_dur <= _durations[l] <= 1.5 * _med_dur]
    if valid_cands:
        rep_ln = int(uniq_ln[valid_cands[int(np.argmax(ln_cnt[valid_cands]))]])
    else:
        rep_ln = int(uniq_ln[int(np.argmax(ln_cnt))])
    rep_mask = (ln == rep_ln)
    if int(np.sum(rep_mask)) < 200:
        rep_mask = np.ones_like(ln, dtype=bool)

    x_m = x_m[rep_mask]
    y_m = y_m[rep_mask]

    # --- Smooth raw GPS a bit before simplification ---
    def _smooth1d(a, w=7, circular=True):
        # Moving-average smoothing without endpoint artifacts.
        # For track loops, circular=True smooths across start/finish instead of pulling ends toward 0.
        w = int(max(3, w))
        if w % 2 == 0:
            w += 1
        n = len(a)
        if n < w:
            return a
        k = np.ones(w, dtype=np.float64) / float(w)
        if circular:
            r = w // 2
            ap = np.concatenate([a[-r:], a, a[:r]])
            sm = np.convolve(ap, k, mode="valid")  # length n + r + r - w + 1 = n
            return sm
        # Non-circular: pad with edge values (no zero-padding kink)
        r = w // 2
        ap = np.pad(a, (r, r), mode="edge")
        sm = np.convolve(ap, k, mode="valid")
        return sm

    x_m = _smooth1d(x_m, w=int(MINIMAP_SMOOTH_WINDOW))
    y_m = _smooth1d(y_m, w=int(MINIMAP_SMOOTH_WINDOW))

    # Cap point count before expensive work
    if len(x_m) > 3000:
        stride = int(np.ceil(len(x_m) / 3000.0))
        x_m = x_m[::stride]
        y_m = y_m[::stride]

    pts_m = np.column_stack([x_m, y_m])

    # --- Remove obvious discontinuities / blips (common near start/finish) ---
    # Instead of hard-splitting (which can create gaps if GPS sampling is sparse),
    # detect "impossibly long" segments relative to the median segment length and
    # drop the outlier point(s) that cause those jump-lines.
    if len(pts_m) >= 50:
        pts_m = np.asarray(pts_m, dtype=np.float64)

        for _pass in range(2):
            d = np.linalg.norm(np.diff(pts_m, axis=0), axis=1)
            if len(d) < 5:
                break

            med = float(np.median(d))
            dyn_thresh = max(float(MINIMAP_JUMP_THRESH_M), med * 6.0)  # robust to sparse sampling
            jump = np.where(d > dyn_thresh)[0]
            if len(jump) == 0:
                break

            keep = np.ones(len(pts_m), dtype=bool)

            # If segment i->i+1 is a jump, usually point i+1 is the outlier (brief pit/paddock/GPS glitch).
            keep[jump + 1] = False

            # If a point has BOTH adjacent segments as jumps, it's almost certainly a bad spike.
            # Mark the middle point out as well.
            for j in jump:
                mid = j
                if 0 < mid < len(pts_m) - 1:
                    left = np.linalg.norm(pts_m[mid] - pts_m[mid - 1])
                    right = np.linalg.norm(pts_m[mid + 1] - pts_m[mid])
                    if left > dyn_thresh and right > dyn_thresh:
                        keep[mid] = False

            pts_m = pts_m[keep]

            # If we removed too much, stop (better to keep a small blip than delete the track)
            if len(pts_m) < 50:
                break

    # Ramer–Douglas–Peucker simplification (reduces tiny wiggles)
    def _rdp(points, eps):
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            return pts
        start = pts[0]
        end = pts[-1]
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-12:
            d = np.linalg.norm(pts - start, axis=1)
        else:
            t = np.clip(((pts - start) @ seg) / seg_len2, 0.0, 1.0)
            proj = start + np.outer(t, seg)
            d = np.linalg.norm(pts - proj, axis=1)
        idx = int(np.argmax(d))
        if float(d[idx]) <= float(eps):
            return np.vstack([start, end])
        left = _rdp(pts[: idx + 1], eps)
        right = _rdp(pts[idx:], eps)
        return np.vstack([left[:-1], right])

    pts_m = _rdp(pts_m, eps=float(MINIMAP_SIMPLIFY_EPS_M))

    # Close loop if endpoints are close
    if len(pts_m) >= 10 and np.linalg.norm(pts_m[0] - pts_m[-1]) < 10.0:
        pts_m = np.vstack([pts_m, pts_m[0]])

    # --- Corner smoothing (Chaikin) to make corners less sharp ---
    def _chaikin(pts, iters=2):
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 4:
            return pts
        closed = np.linalg.norm(pts[0] - pts[-1]) < 6.0
        if closed:
            pts_work = pts[:-1]
        else:
            pts_work = pts
        for _ in range(int(max(0, iters))):
            new_pts = []
            n = len(pts_work)
            for i in range(n - (0 if closed else 1)):
                p0 = pts_work[i]
                p1 = pts_work[(i + 1) % n] if closed else pts_work[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.append(q)
                new_pts.append(r)
            if not closed:
                # keep endpoints for open polyline
                new_pts.insert(0, pts_work[0])
                new_pts.append(pts_work[-1])
            pts_work = np.asarray(new_pts, dtype=np.float64)
        if closed:
            pts_work = np.vstack([pts_work, pts_work[0]])
        return pts_work

    pts_m = _chaikin(pts_m, iters=int(MINIMAP_CORNER_SMOOTH_ITERS))

    # Cap again
    if len(pts_m) > 5000:
        stride = int(np.ceil(len(pts_m) / 5000.0))
        pts_m = pts_m[::stride]

    # --- Start/finish offset (roll centerline by meters) ---
    if 'MINIMAP_SF_SHIFT_M' in globals() and MINIMAP_SF_SHIFT_M and abs(float(MINIMAP_SF_SHIFT_M)) > 1e-6:
        if len(pts_m) > 1 and np.linalg.norm(pts_m[0] - pts_m[-1]) < 1e-6:
            pts_m = pts_m[:-1]
        dxy = np.diff(pts_m, axis=0, append=pts_m[:1])
        seg = np.hypot(dxy[:, 0], dxy[:, 1])
        cum = np.cumsum(seg)
        total = float(cum[-1]) if len(cum) else 0.0
        if total > 1e-6:
            shift = float(MINIMAP_SF_SHIFT_M) % total
            idx = int(np.searchsorted(cum, shift, side='left')) % len(pts_m)
            if idx != 0:
                pts_m = np.roll(pts_m, -idx, axis=0)
        pts_m = np.vstack([pts_m, pts_m[0]])

    # Bounds for transform
    x = pts_m[:, 0]
    y = pts_m[:, 1]
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if (x_max - x_min) < 1e-6 or (y_max - y_min) < 1e-6:
        print("Minimap: degenerate GPS bounds -> disabled.")
        return

    w = int(round(MINIMAP_W_BASE * UI_SCALE))
    h = int(round(MINIMAP_H_BASE * UI_SCALE))
    pad = int(round(MINIMAP_PAD_BASE * UI_SCALE))

    dx = x_max - x_min
    dy = y_max - y_min
    s = min((w - 2 * pad) / dx, (h - 2 * pad) / dy)

    # Center the whole map in the square (letterbox/pillarbox as needed)
    content_w = dx * s
    content_h = dy * s
    x_off = (w - content_w) * 0.5
    y_off = (h - content_h) * 0.5

    def to_px(xv, yv):
        px = int(round(x_off + (xv - x_min) * s))
        py = int(round(h - y_off - (yv - y_min) * s))
        return px, py

    pts_px = np.array([to_px(float(xi), float(yi)) for xi, yi in zip(x, y)], dtype=np.int32)

    base = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(base, (0, 0), (w - 1, h - 1), (20, 20, 20), thickness=-1)
    cv2.rectangle(base, (0, 0), (w - 1, h - 1), (80, 80, 80), thickness=max(1, int(round(UI_SCALE))))

    thick = max(1, int(round(MINIMAP_TRACK_THICKNESS_BASE * UI_SCALE)))
    cv2.polylines(base, [pts_px.reshape((-1, 1, 2))], isClosed=True, color=(255, 255, 255),
                  thickness=thick, lineType=cv2.LINE_AA)

    # Start/finish marker: drawn across the racing line at the start point.
    # Robust to MINIMAP_SF_SHIFT_M roll producing duplicate/flat segments.
    # Length is controlled via MINIMAP_SF_HALF_LEN_PX (half-length in px @ UI_SCALE=1).
    if len(pts_px) >= 2:
        p0 = pts_px[0].astype(np.float64)

        # Find a forward direction (skip duplicates near the rolled start).
        dir_vec = None
        for k in range(1, min(50, len(pts_px))):
            d = (pts_px[k].astype(np.float64) - p0)
            n = float(np.hypot(d[0], d[1]))
            if n > 1e-6:
                dir_vec = d / n
                break

        if dir_vec is not None:
            perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float64)
            half_len_px = float(MINIMAP_SF_HALF_LEN_PX) * float(UI_SCALE)
            if DEBUG_PRINTS:
                print(f"Minimap SF: half_len_px={half_len_px:.1f} (base={MINIMAP_SF_HALF_LEN_PX}), UI_SCALE={UI_SCALE:.2f}")

            a = (p0 - perp * half_len_px).astype(np.int32)
            b = (p0 + perp * half_len_px).astype(np.int32)

            sf_th = max(1, int(round(MINIMAP_SF_THICKNESS_BASE * UI_SCALE)))
            cv2.line(base, tuple(a), tuple(b), (0, 0, 0), thickness=sf_th + 2, lineType=cv2.LINE_AA)
            cv2.line(base, tuple(a), tuple(b), (255, 255, 255), thickness=sf_th, lineType=cv2.LINE_AA)

    _minimap_base = base
    _minimap_transform = (lon0, lat0, cos_lat0, x_min, y_min, s, x_off, y_off, w, h)
    _minimap_centerline_m = pts_m
    _minimap_centerline_px = pts_px
    _build_centerline_progress()


# -------------------- Split (delta vs session-best lap at same track progress) --------------------
# Reference lap updates ONLY when a new session-best lap is completed.
_prev_lap_delta_s = np.nan  # last completed lap delta vs session best at the time (seconds)
_split_best_lap_s = np.inf
_split_ref_prog = None  # normalized 0..1
_split_ref_time = None  # seconds since lap start
_split_last_bi = None
_split_base_prog = None  # progress at start of current lap (0..1)
_split_center_prog = None  # per-centerline point progress 0..1
_split_ready = False  # True only after we observe at least one completed lap during rendering
_split_render_start_unix = None  # unix time (RaceChrono domain) at first rendered frame

# --- Progress / dot anti-jump state ---
_prog_last = np.nan
_prog_last_unix = np.nan
_prog_last_xy_m = None  # last accepted projected XY (meters) for cross-track jump gating
_prog_last_accept = True  # whether last GPS projection was accepted (not rejected by anti-jump gate)

# Pending candidate (debounce) for cross-track flips
_prog_pending_p = None
_prog_pending_xy_m = None
_prog_pending_count = 0
_dot_hint_seg = None
_dot_last_proj_m = None

_dot_last_px = None
_dot_smooth_px = None
_dot_last_dir_px = (1.0, 0.0)
_dot_last_prog = None
# --- Virtual splits ---

# Current-lap split state (resets each lap/segment)
_split_next_idx = 1  # next split index to capture (1..N-1)
_split_last_idx = 0  # last captured split index (0 means start)
_split_last_delta_s = np.nan

_split_prev_rel_p = np.nan
_split_prev_t_s = np.nan

# Running split smoothing (prevents jittery +/- spikes when projection snaps between nearby segments)
_split_run_prev_p = np.nan




def _reset_virtual_splits():
    global _split_next_idx, _split_last_idx, _split_last_delta_s, _split_prev_rel_p, _split_prev_t_s, _split_run_prev_p
    _split_next_idx = 1
    _split_last_idx = 0
    _split_last_delta_s = np.nan
    _split_prev_rel_p = np.nan
    _split_prev_t_s = np.nan
    _split_run_prev_p = np.nan


def _update_virtual_splits(rel_p, lap_time_s):
    """Update and return (last_idx, last_delta_s). rel_p in [0,1] and should be monotonic within lap.
    We compute the split *crossing time* by interpolating between the previous and current sample,
    which prevents large bogus deltas when progress jitters near ambiguous map points.
    """
    global _split_next_idx, _split_last_idx, _split_last_delta_s, _split_prev_rel_p, _split_prev_t_s
    if not VIRTUAL_SPLITS_ENABLE or VIRTUAL_SPLITS_N is None or int(VIRTUAL_SPLITS_N) <= 1:
        return 0, np.nan
    if (_split_ref_prog is None) or (_split_ref_time is None) or (not np.isfinite(lap_time_s)) or (
            not np.isfinite(rel_p)):
        return _split_last_idx, _split_last_delta_s

    # initialize previous sample
    if not np.isfinite(_split_prev_rel_p) or not np.isfinite(_split_prev_t_s):
        _split_prev_rel_p = float(rel_p)
        _split_prev_t_s = float(lap_time_s)
        return _split_last_idx, _split_last_delta_s

    N = int(VIRTUAL_SPLITS_N)
    p_prev = float(_split_prev_rel_p)
    t_prev = float(_split_prev_t_s)
    p_now = float(rel_p)
    t_now = float(lap_time_s)

    # Progress should generally increase; tolerate tiny regressions but don't allow huge backwards jumps
    # (those come from snapping to the wrong part of the polyline).
    if p_now + 0.50 < p_prev:
        # likely wrapped or discontinuity; clamp
        p_now = p_prev

    # capture splits at p = k/N for k=1..N-1 (finish line is implicitly p=1.0)
    while _split_next_idx < N:
        p_k = float(_split_next_idx) / float(N)
        if p_now < p_k:
            break

        # estimate the time we crossed p_k (linear interp in (p,t) between previous and current)
        if p_now > p_prev + 1e-9:
            frac = (p_k - p_prev) / (p_now - p_prev)
            frac = float(np.clip(frac, 0.0, 1.0))
            t_cross = t_prev + frac * (t_now - t_prev)
        else:
            # no progress (or regression) between samples: fall back to current time
            t_cross = t_now

        ref_t = float(np.interp(p_k, _split_ref_prog, _split_ref_time))
        _split_last_delta_s = float(t_cross - ref_t)
        _split_last_idx = int(_split_next_idx)
        _split_next_idx += 1

    _split_prev_rel_p = float(p_now)
    _split_prev_t_s = float(t_now)
    return _split_last_idx, _split_last_delta_s

    N = int(VIRTUAL_SPLITS_N)
    # capture splits at p = k/N for k=1..N-1
    # (finish line is implicitly p=1.0)
    while _split_next_idx < N and rel_p >= (_split_next_idx / N):
        p_k = float(_split_next_idx) / float(N)
        ref_t = float(np.interp(p_k, _split_ref_prog, _split_ref_time))
        _split_last_delta_s = float(lap_time_s - ref_t)
        _split_last_idx = int(_split_next_idx)
        _split_next_idx += 1

    return _split_last_idx, _split_last_delta_s



def _filter_running_rel_p(rel_p: float, speed_mps: float, dt_s: float) -> float:
    """Make rel_p (0..1 within lap) monotonic-ish and rate-limited to avoid split delta spikes.
    This only affects the *displayed running split*; lap timing / boundaries remain unchanged.
    """
    global _split_run_prev_p, _split_center_total_m
    if not np.isfinite(rel_p):
        return rel_p
    # init
    if not np.isfinite(_split_run_prev_p):
        _split_run_prev_p = float(rel_p)
        return float(rel_p)
    p_prev = float(_split_run_prev_p)

    # Reject big backwards snaps (projection jumped to earlier segment). Keep monotonic.
    if rel_p < p_prev - 0.02:
        rel_p = p_prev

    # Rate limit forward motion based on speed and track length (with some headroom).
    total_m = float(_split_center_total_m) if (_split_center_total_m is not None and np.isfinite(_split_center_total_m)) else 1000.0
    spd = float(speed_mps) if np.isfinite(speed_mps) else 0.0
    dt = float(dt_s) if np.isfinite(dt_s) and dt_s > 0 else (1.0 / 30.0)

    # expected progress step ~= (distance / total_length)
    max_step = (max(0.0, spd) * dt) / max(1.0, total_m)
    # allow some slack for GPS noise / underreport factor
    max_step = max_step * 3.0 + 0.01

    if rel_p > p_prev + max_step:
        rel_p = p_prev + max_step

    # light smoothing (keeps it responsive but kills 1-frame zigzags)
    alpha = 0.35
    p_f = p_prev + alpha * (rel_p - p_prev)
    p_f = float(np.clip(p_f, 0.0, 1.0))
    _split_run_prev_p = p_f
    return p_f


# For robust progress estimation, we project GPS onto the minimap centerline polyline and track a local
# window around the previous best segment to avoid snapping to the wrong nearby section at hairpins.
_split_center_cum_m = None
_split_center_seglen_m = None
_split_center_total_m = None
_prog_hint_seg = None  # previous best segment index for windowed projection (0..n-2)


def _build_centerline_progress():
    global _split_center_prog, _split_center_cum_m, _split_center_seglen_m, _split_center_total_m, _prog_hint_seg
    if _minimap_centerline_m is None or len(_minimap_centerline_m) < 2:
        _split_center_prog = None
        _split_center_cum_m = None
        _split_center_seglen_m = None
        _split_center_total_m = None
        _prog_hint_seg = None
        return
    pts = np.asarray(_minimap_centerline_m, dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1]) if cum[-1] > 1e-9 else 1.0
    _split_center_cum_m = cum
    _split_center_seglen_m = seglen
    _split_center_total_m = total
    _split_center_prog = (cum / total).astype(np.float64)
    _prog_hint_seg = None


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


def _progress_from_gps(lat_deg, lon_deg, t_unix=None, speed_mps=None):
    """Return filtered progress in [0,1) by projecting GPS onto the minimap centerline.

    When t_unix is provided (runtime), this applies a simple anti-jump gate so progress doesn't
    snap across the track (hairpins/close proximity sections).
    """
    global _prog_hint_seg, _prog_last, _prog_last_unix, _prog_last_xy_m, _prog_last_accept, _prog_pending_p, _prog_pending_xy_m, _prog_pending_count

    if not (np.isfinite(lat_deg) and np.isfinite(lon_deg)):
        return np.nan
    if (_minimap_transform is None) or (_minimap_centerline_m is None) or (_split_center_cum_m is None) or (
            _split_center_total_m is None):
        return np.nan

    lon0, lat0, cos_lat0, x_min, y_min, s, x_off, y_off, w, h = _minimap_transform
    x = (float(lon_deg) - lon0) * 111320.0 * cos_lat0
    y = (float(lat_deg) - lat0) * 110540.0

    # --- Progress projection (windowed) ---
    # Use a *local* window around the last accepted progress to prevent 1-frame flips to nearby parallel segments.
    seg_hint_use = _prog_hint_seg
    win = 180
    if (t_unix is not None) and np.isfinite(t_unix) and np.isfinite(_prog_last) and np.isfinite(_prog_last_unix):
        # Convert last progress to a segment index hint.
        last_dist_m = float(_prog_last) * float(_split_center_total_m)
        seg_hint_use = int(np.searchsorted(_split_center_cum_m, last_dist_m, side='right') - 1)
        seg_hint_use = int(np.clip(seg_hint_use, 0, len(_minimap_centerline_m) - 2))

        # Dynamic window: based on distance we could have traveled since last sample.
        if (speed_mps is None) or (not np.isfinite(speed_mps)):
            speed_mps_eff = 25.0
        else:
            speed_mps_eff = float(np.clip(speed_mps, 0.0, 120.0))
        dt_local = max(0.0, float(t_unix) - float(_prog_last_unix))
        max_dist_m = speed_mps_eff * dt_local * 2.0 + 25.0  # generous margin
        # Convert meters to "segments" using a robust typical segment length.
        seglen_typ = float(np.median(_split_center_seglen_m)) if (_split_center_seglen_m is not None) else 2.0
        seglen_typ = max(0.5, seglen_typ)
        win = int(np.clip(max_dist_m / seglen_typ, 25, 90))

    proj, seg_idx, t = _project_point_to_polyline_windowed_m((x, y), _minimap_centerline_m,
                                                             seg_hint=seg_hint_use, window=win)
    # Only advance the hint if we accept this sample.
    cand_hint = seg_idx
    cand_xy_m = np.array([float(proj[0]), float(proj[1])], dtype=np.float64)

    dist_m = float(_split_center_cum_m[seg_idx] + t * _split_center_seglen_m[seg_idx])
    prog = dist_m / float(_split_center_total_m)
    prog = prog - math.floor(prog)  # [0,1)

    # Anti-jump gate (runtime only)
    if (t_unix is not None) and np.isfinite(t_unix) and np.isfinite(_prog_last) and np.isfinite(_prog_last_unix):
        dt = float(t_unix - _prog_last_unix)
        if dt > 1e-6:
            # Expected max forward progress based on speed and dt (with margin).
            if (speed_mps is None) or (not np.isfinite(speed_mps)):
                speed_mps_eff = 25.0  # fallback ~56 mph
            else:
                speed_mps_eff = float(np.clip(speed_mps, 0.0, 120.0))
            # Cross-track (XY) jump gate: reject candidate if it teleports laterally to a nearby segment.
            if (_prog_last_xy_m is not None) and np.isfinite(_prog_last_xy_m[0]) and np.isfinite(_prog_last_xy_m[1]):
                dx = float(cand_xy_m[0] - _prog_last_xy_m[0])
                dy = float(cand_xy_m[1] - _prog_last_xy_m[1])
                jump_m = float(math.hypot(dx, dy))

                # Compute lateral displacement relative to the local track tangent at last progress.
                # This catches "parallel segment flips" even when along-track distance is plausible.
                try:
                    last_dist_m = float(_prog_last) * float(_split_center_total_m)
                    seg = int(np.searchsorted(_split_center_cum_m, last_dist_m, side='right') - 1)
                    seg = int(np.clip(seg, 0, len(_minimap_centerline_m) - 2))
                    pA = _minimap_centerline_m[seg]
                    pB = _minimap_centerline_m[seg + 1]
                    tx = float(pB[0] - pA[0])
                    ty = float(pB[1] - pA[1])
                    tn = float(math.hypot(tx, ty))
                    if tn > 1e-9:
                        tx /= tn
                        ty /= tn
                        along = dx * tx + dy * ty
                        lat = float(math.sqrt(max(0.0, jump_m * jump_m - along * along)))
                    else:
                        lat = jump_m
                except Exception:
                    lat = jump_m

                # Allow some slack beyond physical travel; mainly catches lateral teleports.
                max_jump_m = float(speed_mps_eff * dt * 1.2 + 4.0)
                max_lat_m  = float(speed_mps_eff * dt * 0.6 + 3.5)
                # Hard cap lateral moves to avoid parallel-segment flips.
                max_lat_m = min(max_lat_m, float(MINIMAP_MAX_LATERAL_JUMP_M))

                # Debounce: single-frame flips can pass the physical gate at speed. If the candidate implies a
                # significant cross-track move, require it to be stable for a few consecutive frames before accepting.
                confirm_jump_m = float(MINIMAP_CONFIRM_JUMP_M)
                confirm_lat_m = float(MINIMAP_CONFIRM_LATERAL_M)
                confirm_streak = int(MINIMAP_CONFIRM_STREAK)

                if (jump_m > confirm_jump_m) or (lat > confirm_lat_m):
                    same_pending = False
                    if (_prog_pending_xy_m is not None) and np.isfinite(_prog_pending_xy_m[0]) and np.isfinite(_prog_pending_xy_m[1]):
                        pdx = float(cand_xy_m[0] - _prog_pending_xy_m[0])
                        pdy = float(cand_xy_m[1] - _prog_pending_xy_m[1])
                        if math.hypot(pdx, pdy) < 3.0:
                            same_pending = True

                    if (not same_pending):
                        _prog_pending_xy_m = cand_xy_m.copy()
                        _prog_pending_p = float(prog)
                        _prog_pending_count = 1
                    else:
                        _prog_pending_count = int(_prog_pending_count) + 1

                    if _prog_pending_count < confirm_streak:
                        # Reject for now: smoothly advance from last progress (prevents flicker).
                        pred_step = (speed_mps_eff * dt) / float(_split_center_total_m)
                        pred_step = float(np.clip(pred_step, 0.0, 0.06))
                        _prog_last = float((_prog_last + pred_step) % 1.0)
                        _prog_last_unix = float(t_unix)
                        _prog_last_accept = False
                        return float(_prog_last)

                    # Stable enough: accept and clear pending state.
                    _prog_pending_count = 0
                    _prog_pending_p = None
                    _prog_pending_xy_m = None
                else:
                    # No significant lateral move: clear pending.
                    _prog_pending_count = 0
                    _prog_pending_p = None
                    _prog_pending_xy_m = None

                if (jump_m > max_jump_m) or (lat > max_lat_m):
                    # Reject: smoothly advance from last progress (prevents freeze + snapback).
                    pred_step = (speed_mps_eff * dt) / float(_split_center_total_m) if (_split_center_total_m and _split_center_total_m > 0) else 0.0
                    pred_step = float(np.clip(pred_step, 0.0, 0.06))
                    _prog_last = float((_prog_last + pred_step) % 1.0)
                    _prog_last_unix = float(t_unix)
                    _prog_last_accept = False
                    return float(_prog_last)

            max_fwd = (speed_mps_eff * dt) / float(_split_center_total_m) + 0.04
            max_fwd = float(np.clip(max_fwd, 0.01, 0.25))

            # signed shortest delta in (-0.5, 0.5]
            d = ((prog - _prog_last + 0.5) % 1.0) - 0.5

            # Don't allow large backward jumps, and don't allow implausibly large forward jumps.
            if (d < -0.03) or (d > max_fwd):
                # Reject: advance progress smoothly instead of freezing (prevents spike-then-snapback).
                if (t_unix is not None) and np.isfinite(t_unix) and (_prog_last_unix is not None) and np.isfinite(
                        _prog_last_unix):
                    dt = max(0.0, float(t_unix) - float(_prog_last_unix))
                    # If speed is provided, predict forward travel; otherwise just cap to max_fwd.
                    if speed_mps is not None and np.isfinite(speed_mps) and (
                            _split_center_total_m is not None) and np.isfinite(
                            _split_center_total_m) and _split_center_total_m > 0:
                        pred_step = (float(speed_mps) * dt) / float(_split_center_total_m)
                        pred_step = min(float(max_fwd), max(0.0, pred_step))
                    else:
                        pred_step = float(max_fwd) * 0.5  # conservative nudge
                    _prog_last = float((_prog_last + pred_step) % 1.0)
                    _prog_last_unix = float(t_unix)
                    _prog_last_accept = False
                    return float(_prog_last)
                _prog_last_accept = False
                return float(_prog_last)

    # Accept sample
    _prog_last_accept = True
    _prog_hint_seg = cand_hint
    if (t_unix is not None) and np.isfinite(t_unix):
        _prog_last = float(prog)
        _prog_last_unix = float(t_unix)
        _prog_last_xy_m = cand_xy_m
    return float(prog)



# --- Minimap dot: RaceChrono-only progress with short predictive hold ---
# Uses RaceChrono GPS projection for progress (same basis as reference lap), but when GPS projection is temporarily rejected
# (close-parallel flips, multipath), it will predict forward briefly using speed to avoid a visible freeze, while clamping
# how far it can run ahead.
MINIMAP_USE_RACECHRONO_ONLY = True
MINIMAP_RC_HOLD_PREDICT_S = 0.60   # seconds to allow prediction after last accepted GPS projection
MINIMAP_RC_MAX_AHEAD_M = 8.0       # max distance dot may run ahead of last accepted GPS position during hold
MINIMAP_RC_BLEND_ALPHA = 0.35      # blend toward GPS when accepted (0..1)
MINIMAP_RC_MAX_BACKSTEP_FRAC = 0.0015  # allow tiny backstep (fraction of lap) to reduce jitter; larger backsteps rejected
MINIMAP_RC_MAX_FWD_FRAC_PER_S = 0.20   # limit forward progress change rate (fraction of lap per second) for dot stability

_minimap_rc_last_accept_p = np.nan
_minimap_rc_last_accept_unix = np.nan
_minimap_rc_disp_p = np.nan
# --- Fused progress (RaceChrono GPS + Haltech speed dead-reckon) ---
# This is designed for hairpins/close-parallel sections where GPS projection can "flip" to a nearby segment for 1-2 frames,
# causing split delta spikes and minimap dot freezes/jumps.
FUSED_UNDERREPORT_SPEED = 1.00  # 1.0 = no intentional drift; rely on GPS blend for correction
FUSED_GPS_BLEND_ALPHA = 0.18  # 0..1, how strongly to pull prediction toward GPS when GPS is sane
FUSED_MAX_PULLBACK_FRAC = 0.0015  # never pull back more than this (fraction of lap), else ignore GPS
FUSED_MAX_SNAP_FWD_FRAC = 0.003  # limit how far ahead GPS can snap us in one update (fraction of lap)
# --- GPS sanity gating (prevents early teleports to wrong nearby segment) ---
FUSED_GPS_SANE_WINDOW_FRAC = 0.003  # GPS must be within this fraction of lap from prediction to be trusted
FUSED_GPS_SANE_STREAK = 3  # consecutive sane samples required before GPS blending resumes (lower = less lag after SF)

# --- Centerline enable gating (minimap handoff) ---
FUSED_CENTERLINE_ENABLE_WINDOW_FRAC = 0.0012  # must be within this fraction of lap to enable centerline drawing
FUSED_CENTERLINE_ENABLE_STREAK = 15  # consecutive samples required to enable centerline drawing

FUSED_REQUIRE_GPS_LOCK_S = 2.0  # seconds of sane GPS progress change required before enabling dead-reckon
FUSED_LOCK_MIN_SPEED_MPS = 3.0  # require at least this speed while locking (filters pit creep)
FUSED_LOCK_MIN_DP_FRAC = 0.00035  # minimum progress change per sample to count as "moving" while locking

# After GPS lock is achieved, ignore GPS for this many seconds and advance only by speed prediction.
# Prevents early on-track segment flips (teleports) while GPS/projection stabilizes.
FUSED_POSTLOCK_IGNORE_GPS_S = 6.0

_fused_prog = None
_fused_unix = None

_fused_gps_sane_count = 0
_fused_centerline_ok_count = 0
_fused_locked = False
_fused_lock_t0 = None
_fused_last_gps_p = None


def _unwrap_near(x, ref):
    """Return x (possibly +/-1) that is closest to ref."""
    cands = (x, x + 1.0, x - 1.0)
    return min(cands, key=lambda v: abs(v - ref))


def _progress_fused(lat_deg, lon_deg, t_unix=None, speed_mps=None):
    """Progress estimator in [0,1). Uses speed dead-reckon only AFTER a short GPS lock period.
    Lock prevents wrong initial position / pit creep. After lock, GPS is blended forward-only (no pullbacks).
    """
    global _fused_prog, _fused_unix, _fused_locked, _fused_lock_t0, _fused_last_gps_p, _fused_gps_sane_count, _fused_centerline_ok_count

    # If no valid timebase, fall back to GPS-only.
    if (t_unix is None) or (not np.isfinite(t_unix)):
        p = _progress_from_gps(lat_deg, lon_deg, t_unix=t_unix, speed_mps=speed_mps)
        return float(p % 1.0) if np.isfinite(p) else 0.0

    t_unix = float(t_unix)
    spd = float(speed_mps) if (speed_mps is not None and np.isfinite(speed_mps)) else 0.0

    gps_p = _progress_from_gps(lat_deg, lon_deg, t_unix=t_unix, speed_mps=speed_mps)
    gps_ok = np.isfinite(gps_p)
    gps_p = float(gps_p % 1.0) if gps_ok else None

    # Seed from first valid GPS progress (or 0).
    if _fused_prog is None or _fused_unix is None:
        _fused_prog = float(gps_p) if gps_ok else 0.0
        _fused_unix = t_unix
        _fused_locked = False
        _fused_lock_t0 = None
        _fused_last_gps_p = float(gps_p) if gps_ok else None
        return float(_fused_prog)

    # --- Locking phase: stay glued to GPS until it is clearly moving on-track ---
    if not _fused_locked:
        moving = False
        if gps_ok and (spd >= float(FUSED_LOCK_MIN_SPEED_MPS)) and (_fused_last_gps_p is not None):
            prev = float(_fused_last_gps_p)
            cur_u = _unwrap_near(float(gps_p), prev)
            dp = abs(cur_u - prev)
            if dp >= float(FUSED_LOCK_MIN_DP_FRAC):
                moving = True

        if moving:
            if _fused_lock_t0 is None:
                _fused_lock_t0 = t_unix
            elif (t_unix - float(_fused_lock_t0)) >= float(FUSED_REQUIRE_GPS_LOCK_S):
                _fused_locked = True
        else:
            _fused_lock_t0 = None

        if gps_ok:
            _fused_prog = float(gps_p)
            _fused_last_gps_p = float(gps_p)
        _fused_unix = t_unix
        _fused_centerline_ok_count = 0
        return float(_fused_prog)

    # --- Locked phase: dead-reckon forward, blend GPS forward-only ---
    dt = max(0.0, t_unix - float(_fused_unix))
    _fused_unix = t_unix

    # Predict forward using speed (slightly under-reported)
    step = (spd * dt) / float(_split_center_total_m) if _split_center_total_m > 1e-9 else 0.0
    step *= float(FUSED_UNDERREPORT_SPEED)
    pred = float((_fused_prog + step) % 1.0)

    # After lock: for a short window, ignore GPS and use speed-only prediction.
    # This avoids early teleports from GPS projection flipping to a nearby segment.
    if _fused_locked and (_fused_lock_t0 is not None):
        if (t_unix - _fused_lock_t0) < float(FUSED_POSTLOCK_IGNORE_GPS_S):
            _fused_prog = pred
            return float(_fused_prog)

    if not gps_ok:
        _fused_prog = pred
        return float(_fused_prog)

    gps_u = _unwrap_near(float(gps_p), pred)

    # GPS sanity gating: only trust GPS when it is close to our speed-based prediction for several samples.
    d_pred = abs(gps_u - pred)
    if d_pred <= float(FUSED_GPS_SANE_WINDOW_FRAC):
        _fused_gps_sane_count += 1
    else:
        _fused_gps_sane_count = 0

    # Separate (stricter/longer) streak used to ENABLE centerline mapping in the minimap.
    if d_pred <= float(FUSED_CENTERLINE_ENABLE_WINDOW_FRAC):
        _fused_centerline_ok_count += 1
    else:
        _fused_centerline_ok_count = 0

    if _fused_gps_sane_count < int(FUSED_GPS_SANE_STREAK):
        _fused_prog = pred
        return float(_fused_prog)

    max_pullback = float(FUSED_MAX_PULLBACK_FRAC)
    max_snap_fwd = float(FUSED_MAX_SNAP_FWD_FRAC)

    # Never let GPS pull us backward significantly (hairpin flip). Ignore if so.
    if gps_u < (pred - max_pullback):
        _fused_prog = pred
        _fused_last_gps_p = float(gps_p)
        return float(_fused_prog)

    # Limit forward snap in one update
    gps_u = min(gps_u, pred + max_snap_fwd)

    a = float(FUSED_GPS_BLEND_ALPHA)
    fused = pred + a * (gps_u - pred)
    _fused_prog = float(fused % 1.0)
    _fused_last_gps_p = float(gps_p)
    return float(_fused_prog)


def _progress_minimap_rc_hold(lat_deg, lon_deg, t_unix, speed_mps):
    """Progress used for minimap dot.
    If MINIMAP_USE_RACECHRONO_ONLY is True, use RaceChrono GPS projection with anti-jump gates,
    but predict forward briefly when GPS projection is rejected to avoid visible freezes.
    """
    global _minimap_rc_last_accept_p, _minimap_rc_last_accept_unix, _minimap_rc_disp_p, _minimap_rc_last_accept_u, _minimap_rc_disp_u

    if (t_unix is None) or (not np.isfinite(t_unix)):
        return np.nan

    # If not RC-only, fall back to fused progress (legacy behavior).
    if not MINIMAP_USE_RACECHRONO_ONLY:
        return _progress_fused(lat_deg, lon_deg, t_unix=t_unix, speed_mps=speed_mps)

    # Compute GPS-only projected progress with anti-jump/XY gates. _prog_last_accept tells whether it was accepted.
    p_gps = _progress_from_gps(lat_deg, lon_deg, t_unix=t_unix, speed_mps=speed_mps)
    gps_ok = bool(_prog_last_accept) and np.isfinite(p_gps)

    if (not np.isfinite(_minimap_rc_disp_p)) or (not np.isfinite(_minimap_rc_last_accept_unix)) or (not np.isfinite(_minimap_rc_disp_u)) or (not np.isfinite(_minimap_rc_last_accept_u)):
        # Initialize state on first valid sample
        if np.isfinite(p_gps):
            _minimap_rc_disp_p = float(p_gps)
            _minimap_rc_last_accept_p = float(p_gps) if gps_ok else np.nan
            _minimap_rc_last_accept_unix = float(t_unix)
            _minimap_rc_last_accept_u = float(p_gps) if gps_ok else float(p_gps)
            _minimap_rc_disp_u = float(p_gps)
            return float(_minimap_rc_disp_p)
        return np.nan

    dt = float(t_unix - float(_minimap_rc_last_accept_unix if gps_ok else _minimap_rc_last_accept_unix))
    dt = max(0.0, min(dt, 1.0))

    lap_len_m = float(_split_center_total_m) if (_split_center_total_m and np.isfinite(_split_center_total_m) and _split_center_total_m > 0) else 1.0

    # --- unwrap candidate progress to avoid 1-frame wrap blips ---
    def _unwrap_prog(cand_p_wrapped, last_u, exp_fwd_frac):
        if (last_u is None) or (not np.isfinite(last_u)) or (not np.isfinite(cand_p_wrapped)):
            return float(cand_p_wrapped)
        base = math.floor(last_u)
        target = last_u + max(0.0, float(exp_fwd_frac))
        best_u = None
        best_err = 1e18
        for k in (-1, 0, 1):
            u = float(cand_p_wrapped) + base + k
            err = abs(u - target)
            if err < best_err:
                best_err = err
                best_u = u
        # prevent big backward jumps
        if best_u < last_u - 0.05:
            best_u = last_u - 0.05
        return float(best_u)
    max_ahead_frac = float(MINIMAP_RC_MAX_AHEAD_M) / lap_len_m

    def shortest_delta(a, b):
        return ((a - b + 0.5) % 1.0) - 0.5
    exp_fwd_frac = 0.0
    if (speed_mps is not None) and np.isfinite(speed_mps) and (lap_len_m > 1e-3):
        exp_fwd_frac = float(speed_mps) * float(dt) / float(lap_len_m)

    if gps_ok:
        cand_u = _unwrap_prog(float(p_gps), _minimap_rc_last_accept_u, exp_fwd_frac)

        # Blend display toward candidate (unwrapped), with stability constraints.
        if not np.isfinite(_minimap_rc_disp_u):
            _minimap_rc_disp_u = float(cand_u)
        d = float(cand_u - float(_minimap_rc_disp_u))

        # allow tiny backstep; clamp large backsteps
        if d < -float(MINIMAP_RC_MAX_BACKSTEP_FRAC):
            d = -float(MINIMAP_RC_MAX_BACKSTEP_FRAC)

        # limit forward rate
        max_fwd = float(MINIMAP_RC_MAX_FWD_FRAC_PER_S) * max(1e-3, float(dt))
        if d > max_fwd:
            d = max_fwd

        # low-pass blend
        alpha = float(MINIMAP_RC_BLEND_ALPHA)
        _minimap_rc_disp_u = float(_minimap_rc_disp_u) + alpha * d

        # update accept state
        _minimap_rc_last_accept_u = float(cand_u)
        _minimap_rc_last_accept_p = float(p_gps)
        _minimap_rc_last_accept_unix = float(t_unix)

        _minimap_rc_disp_p = float(_minimap_rc_disp_u % 1.0)
        return float(_minimap_rc_disp_p)

    age = float(t_unix - float(_minimap_rc_last_accept_unix))
    if age <= float(MINIMAP_RC_HOLD_PREDICT_S):
        if (speed_mps is None) or (not np.isfinite(speed_mps)):
            speed_eff = 0.0
        else:
            speed_eff = float(np.clip(speed_mps, 0.0, 120.0))
        step_u = 0.0
        if lap_len_m > 1e-3:
            step_u = (speed_eff * dt) / lap_len_m
        step_u = float(np.clip(step_u, 0.0, float(MINIMAP_RC_MAX_FWD_FRAC_PER_S) * max(1e-3, dt)))

        if np.isfinite(_minimap_rc_disp_u):
            pred_u = float(_minimap_rc_disp_u + step_u)
        else:
            pred_u = float(_minimap_rc_last_accept_u + step_u) if np.isfinite(_minimap_rc_last_accept_u) else float(step_u)

        # clamp so we don't run too far ahead of last accepted GPS progress (in meters -> progress)
        if np.isfinite(_minimap_rc_last_accept_u):
            max_ahead_u = float(max_ahead_frac)
            if pred_u > float(_minimap_rc_last_accept_u) + max_ahead_u:
                pred_u = float(_minimap_rc_last_accept_u) + max_ahead_u

        _minimap_rc_disp_u = float(pred_u)
        _minimap_rc_disp_p = float(_minimap_rc_disp_u % 1.0)
        return float(_minimap_rc_disp_p)

    # Too stale: hold last display progress
    return float(_minimap_rc_disp_p)



def _slice_lap_samples(t0, t1):
    # returns indices in lap_t for [t0, t1]
    i0 = int(np.searchsorted(lap_t, t0, side="left"))
    i1 = int(np.searchsorted(lap_t, t1, side="right"))
    return i0, i1


def _make_split_reference(t0, t1):
    global _split_ref_prog, _split_ref_time
    if (lap_gps_lat is None) or (lap_gps_lon is None):
        _split_ref_prog, _split_ref_time = None, None
        return
    i0, i1 = _slice_lap_samples(t0, t1)
    if (i1 - i0) < 10:
        _split_ref_prog, _split_ref_time = None, None
        return
    ts = lap_t[i0:i1]
    lats = lap_gps_lat[i0:i1]
    lons = lap_gps_lon[i0:i1]
    prog = np.array([_progress_from_gps(a, o) for a, o in zip(lats, lons)], dtype=np.float64)
    m = np.isfinite(prog) & np.isfinite(ts)
    prog = prog[m]
    ts = ts[m]
    if len(prog) < 10:
        _split_ref_prog, _split_ref_time = None, None
        return
    # unwrap progress relative to first point
    base = float(prog[0])
    rel = prog - base
    rel[rel < 0.0] += 1.0
    # normalize to [0,1] using last progress (avoid small drift)
    end = float(np.nanmax(rel))
    if end < 1e-3:
        _split_ref_prog, _split_ref_time = None, None
        return
    ref_p = rel / end
    ref_t = ts - float(ts[0])
    # enforce monotonic progress (drop backward points)
    keep = np.ones(len(ref_p), dtype=bool)
    last = -1e9
    for k in range(len(ref_p)):
        if ref_p[k] + 1e-6 >= last:
            last = ref_p[k]
        else:
            keep[k] = False
    ref_p = ref_p[keep]
    ref_t = ref_t[keep]
    if len(ref_p) < 10:
        _split_ref_prog, _split_ref_time = None, None
        return
    # ensure endpoints include 0 and 1
    if ref_p[0] > 1e-4:
        ref_p = np.insert(ref_p, 0, 0.0)
        ref_t = np.insert(ref_t, 0, 0.0)
    if ref_p[-1] < 0.999:
        ref_p = np.append(ref_p, 1.0)
        ref_t = np.append(ref_t, float(ref_t[-1]))
    _split_ref_prog, _split_ref_time = ref_p, ref_t


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


def _minimap_dot_xy(lat_deg, lon_deg):
    """Return pixel coords for the minimap dot.
    Uses a windowed projection with hysteresis so the dot doesn't jump to nearby track sections.
    """
    global _dot_hint_seg, _dot_last_proj_m, _dot_last_prog, _dot_last_px, _dot_smooth_px, _dot_last_dir_px
    if _minimap_transform is None:
        return None
    lon0, lat0, cos_lat0, x_min, y_min, s, x_off, y_off, w, h = _minimap_transform
    if not (np.isfinite(lat_deg) and np.isfinite(lon_deg)):
        return None

    # Convert current GPS to meters in the same coordinate space as the polyline
    x = (float(lon_deg) - lon0) * 111320.0 * cos_lat0
    y = (float(lat_deg) - lat0) * 110540.0

    if MINIMAP_SNAP_DOT_TO_LINE and (_minimap_centerline_m is not None) and (len(_minimap_centerline_m) >= 2):
        # Windowed search around the last good segment to prevent snapping to a nearby part of the track.
        proj, seg_idx, t = _project_point_to_polyline_windowed_m((x, y), _minimap_centerline_m, seg_hint=_dot_hint_seg,
                                                                 window=180)

        # Hard gate: if the projection moved an implausible distance in one frame, freeze the dot (prevents visible jumps).
        # Also gate on along-track progress to avoid snapping across close-by track segments (hairpins / S-F).
        if _dot_last_proj_m is not None:
            d = float(np.hypot(proj[0] - _dot_last_proj_m[0], proj[1] - _dot_last_proj_m[1]))
            # Along-track progress gating (cyclic)
            prog_ok = True
            try:
                if (_split_center_cum_m is not None) and (_split_center_seglen_m is not None) and (
                        _split_center_total_m is not None):
                    prog_new = float((_split_center_cum_m[seg_idx] + float(t) * _split_center_seglen_m[
                        seg_idx]) / _split_center_total_m)
                else:
                    prog_new = None
            except Exception:
                prog_new = None

            if prog_new is not None and _dot_last_prog is not None:
                dp = abs(prog_new - float(_dot_last_prog))
                dp = min(dp, 1.0 - dp)  # cyclic distance
                if dp > float(MINIMAP_DOT_MAX_STEP_PROG):
                    prog_ok = False

            if (d > float(MINIMAP_DOT_MAX_JUMP_M)) or (not prog_ok):
                # Instead of freezing the dot (stop-then-jump), move toward the new projection by a clamped step
                # AND commit that intermediate position so subsequent rejected samples continue moving smoothly.
                try:
                    dx = float(proj[0] - _dot_last_proj_m[0])
                    dy = float(proj[1] - _dot_last_proj_m[1])
                    dist = float(np.hypot(dx, dy))
                    if dist > 1e-6:
                        step = min(dist, float(MINIMAP_DOT_MAX_JUMP_M))
                        proj = (_dot_last_proj_m[0] + dx / dist * step,
                                _dot_last_proj_m[1] + dy / dist * step)
                    else:
                        proj = _dot_last_proj_m
                except Exception:
                    proj = _dot_last_proj_m

                # Commit the clamped step so the dot doesn't "pause" on consecutive rejected samples.
                _dot_last_proj_m = proj
                # Keep hint/prog unchanged on rejected samples to avoid snapping to a nearby segment.
            else:
                _dot_last_proj_m = proj
                _dot_hint_seg = seg_idx
                _dot_last_prog = prog_new if prog_new is not None else _dot_last_prog
        else:
            _dot_last_proj_m = proj
            _dot_hint_seg = seg_idx
            try:
                if (_split_center_cum_m is not None) and (_split_center_seglen_m is not None) and (
                        _split_center_total_m is not None):
                    _dot_last_prog = float((_split_center_cum_m[seg_idx] + float(t) * _split_center_seglen_m[
                        seg_idx]) / _split_center_total_m)
            except Exception:
                pass

        x, y = float(proj[0]), float(proj[1])

    px = int(round(x_off + (x - x_min) * s))
    py = int(round(h - y_off - (y - y_min) * s))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    # --- pixel-space clamp + smoothing (prevents teleports, reduces jitter, avoids stall shake) ---
    _x = float(px)
    _y = float(py)

    if _dot_last_px is None:
        _dot_last_px = (_x, _y)
        _dot_smooth_px = (_x, _y)
        return (int(round(_x)), int(round(_y)))

    dx = _x - float(_dot_last_px[0])
    dy = _y - float(_dot_last_px[1])
    dist = float((dx * dx + dy * dy) ** 0.5)

    max_step = float(MINIMAP_DOT_MAX_PX_STEP)
    if dist > max_step and dist > 1e-6:
        _x = float(_dot_last_px[0]) + dx / dist * max_step
        _y = float(_dot_last_px[1]) + dy / dist * max_step
        dx = _x - float(_dot_last_px[0])
        dy = _y - float(_dot_last_px[1])
        dist = float((dx * dx + dy * dy) ** 0.5)

    # Track last direction; if we are effectively stalled, optionally nudge along it
    if dist > float(MINIMAP_DOT_STALL_EPS_PX):
        _dot_last_dir_px = (dx / dist, dy / dist)
    elif float(MINIMAP_DOT_MIN_PX_STEP) > 0.0:
        _x = float(_dot_last_px[0]) + float(_dot_last_dir_px[0]) * float(MINIMAP_DOT_MIN_PX_STEP)
        _y = float(_dot_last_px[1]) + float(_dot_last_dir_px[1]) * float(MINIMAP_DOT_MIN_PX_STEP)

    # Low-pass filter to suppress 1px "shake"
    if _dot_smooth_px is None:
        _dot_smooth_px = (_x, _y)
    else:
        a = float(MINIMAP_DOT_SMOOTH_ALPHA)
        _dot_smooth_px = (float(_dot_smooth_px[0]) + a * (_x - float(_dot_smooth_px[0])),
                          float(_dot_smooth_px[1]) + a * (_y - float(_dot_smooth_px[1])))

    _dot_last_px = (_x, _y)
    return (int(round(_dot_smooth_px[0])), int(round(_dot_smooth_px[1])))
    dx = _x - float(_dot_last_px[0])
    dy = _y - float(_dot_last_px[1])
    dist = (dx * dx + dy * dy) ** 0.5

    max_step = float(MINIMAP_DOT_MAX_PX_STEP)
    if dist > max_step and dist > 1e-6:
        _x = float(_dot_last_px[0]) + dx / dist * max_step
        _y = float(_dot_last_px[1]) + dy / dist * max_step
    elif float(MINIMAP_DOT_MIN_PX_STEP) > 0.0 and dist < float(MINIMAP_DOT_MIN_PX_STEP) and dist > 1e-6:
        # optional: nudge to avoid apparent pixel "stalls"
        _x = float(_dot_last_px[0]) + dx / dist * float(MINIMAP_DOT_MIN_PX_STEP)
        _y = float(_dot_last_px[1]) + dy / dist * float(MINIMAP_DOT_MIN_PX_STEP)

    _dot_last_px = (_x, _y)
    return (int(round(_x)), int(round(_y)))


def draw_minimap(frame, cur_lat_deg, cur_lon_deg, t_unix=None, speed_mps=None):
    if _minimap_base is None:
        return
    h, w = _minimap_base.shape[:2]
    margin = int(round(MINIMAP_MARGIN_BASE * UI_SCALE))
    # Anchor minimap based on MINIMAP_POS
    pos = (MINIMAP_POS or 'topleft').lower()
    if pos not in ('topleft', 'topright', 'bottomleft', 'bottomright'):
        pos = 'topleft'
    if 'right' in pos:
        x0 = W - w - margin
    else:
        x0 = margin
    if 'bottom' in pos:
        y0 = H - h - margin
    else:
        y0 = margin
    if y0 + h > frame.shape[0] or x0 + w > frame.shape[1]:
        return

    roi = frame[y0:y0 + h, x0:x0 + w]
    cv2.addWeighted(roi, 0.35, _minimap_base, 0.65, 0.0, dst=roi)

    pt = None

    global _minimap_centerline_active
    try:

        prog = _progress_minimap_rc_hold(cur_lat_deg, cur_lon_deg, t_unix=t_unix, speed_mps=speed_mps)

        if (not _fused_locked) or (_fused_centerline_ok_count < int(FUSED_CENTERLINE_ENABLE_STREAK)):
            _minimap_centerline_active = False
            prog = None

    except Exception:

        prog = None

    if prog is not None and np.isfinite(prog) and (_minimap_centerline_px is not None) and (
            len(_minimap_centerline_px) > 1):

        idx_f = float(prog % 1.0) * (len(_minimap_centerline_px) - 1)

        i0 = int(np.floor(idx_f))

        i1 = min(i0 + 1, len(_minimap_centerline_px) - 1)

        tt = idx_f - i0

        x0, y0 = float(_minimap_centerline_px[i0][0]), float(_minimap_centerline_px[i0][1])

        x1, y1 = float(_minimap_centerline_px[i1][0]), float(_minimap_centerline_px[i1][1])

        pt = (x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt)
        # First frame we switch to centerline mapping: reset dot state so it doesn't "teleport"
        if not _minimap_centerline_active:
            _minimap_centerline_active = True
            _dot_last_px = (float(pt[0]), float(pt[1]))
            _dot_smooth_px = (float(pt[0]), float(pt[1]))

    if pt is None:
        pt = _minimap_dot_xy(cur_lat_deg, cur_lon_deg)
    if pt is None:
        return
    r = max(2, int(round(MINIMAP_DOT_R_BASE * UI_SCALE)))
    # OpenCV expects integer pixel centers
    pt_i = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(roi, pt_i, r + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(roi, pt_i, r, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)


_build_minimap_base()

# Video time window in Haltech samples
VIDEO_DUR_S = (max_frames / fps) if (max_frames and fps) else 0.0

t_main_start = MAIN_START_S
t_main_end = t_main_start + VIDEO_DUR_S

# Main speed (mph) in the exact video window (uniform resample)
MAIN_FS = 5.0  # Hz (keep light; enough for matching)
dt = 1.0 / MAIN_FS
main_grid = np.arange(t_main_start, t_main_end, dt, dtype=np.float64)
main_speed_mph = np.interp(
    main_grid,
    t_log,
    (CACHE_SERIES["speed"] / SCALE["speed"]) * KPH_TO_MPH
)

# Lap speed for whole day (uniform grid)
lap_grid = np.arange(lap_t[0], lap_t[-1], dt, dtype=np.float64)
lap_speed_u = _resample_interp(lap_t, lap_speed, lap_grid)

# Find where main window fits into lap day
idx0 = _match_speed_window_fft(lap_speed_u, main_speed_mph)
lap_grid_start = float(lap_grid[idx0])
LAP_SYNC_OFFSET = lap_grid_start - t_main_start  # unix = haltech_sample + offset (unknown tz but offset handles it)

# Precompute lap times (day best + session best)
# Determine lap boundaries from lap_number changes (on the unix axis)
# NaN-safe lap number int conversion (avoids RuntimeWarning when lap_lapnum has NaNs)
lapnum_int = np.full_like(lap_lapnum, -1, dtype=np.int64)
_valid_ln = np.isfinite(lap_lapnum)
lapnum_int[_valid_ln] = np.round(lap_lapnum[_valid_ln]).astype(np.int64)

valid_lap = np.isfinite(lap_lapnum) & (lapnum_int >= 1)
lap_ts = lap_t[valid_lap]
lapnums = lapnum_int[valid_lap]

lap_starts = {}
lap_ends = {}
# start = first occurrence, end = last occurrence
for ln in np.unique(lapnums):
    m = (lapnums == ln)
    if not np.any(m):
        continue
    lap_starts[int(ln)] = float(lap_ts[m][0])
    lap_ends[int(ln)] = float(lap_ts[m][-1])

lap_durations = {ln: (lap_ends[ln] - lap_starts[ln]) for ln in lap_starts if ln in lap_ends}
# Filter obvious junk laps
good_laps = {ln: d for ln, d in lap_durations.items() if np.isfinite(d) and d > 20.0 and d < 600.0}
DAY_BEST_LAP_S_ALL = float(min(good_laps.values())) if good_laps else float("nan")

# Session laps = laps that overlap the synced video window
win_start_unix = t_main_start + LAP_SYNC_OFFSET
win_end_unix = t_main_end + LAP_SYNC_OFFSET

# "Day best" should reflect only laps completed BEFORE this video's start (based on synced lap log time).
past_laps = {ln: d for ln, d in good_laps.items() if (lap_ends.get(ln) is not None and lap_ends[ln] <= win_start_unix)}
DAY_BEST_LAP_S = float(min(past_laps.values())) if past_laps else float("nan")
session_laps = []
for ln, d in good_laps.items():
    s = lap_starts.get(ln)
    e = lap_ends.get(ln)
    if s is None or e is None:
        continue
    if e >= win_start_unix and s <= win_end_unix:
        session_laps.append(d)
SESSION_BEST_LAP_S = float(min(session_laps)) if session_laps else float("nan")

# For "Prev" lap, only count laps that occur in the rendered (trimmed) video window
session_good_laps = {}
for ln, d in good_laps.items():
    s = lap_starts.get(ln)
    e = lap_ends.get(ln)
    if s is None or e is None:
        continue
    if e >= win_start_unix and s <= win_end_unix:
        session_good_laps[int(ln)] = float(d)
FIRST_SESSION_LAPNUM = int(min(session_good_laps.keys())) if session_good_laps else 10 ** 9
# -------------------- Session boundaries within rendered window --------------------
# Define "session laps" strictly within the rendered (trimmed) output window:
# - Session starts at the rendered window start (output t=0), so lap time is 0 at output start.
# - Each time the lap CSV indicates a new lap start within the window, we start a new session-lap segment.
# - The final segment ending at the rendered window end is treated as incomplete (not counted for "Prev" or "Sess best").
_session_starts = []
for _ln, _s in lap_starts.items():
    try:
        _s = float(_s)
    except Exception:
        continue
    if win_start_unix < _s < win_end_unix:
        _session_starts.append(_s)

_session_starts = sorted(set(_session_starts))
session_boundaries_unix = [float(win_start_unix)] + _session_starts + [float(win_end_unix)]

# De-dupe boundaries that are extremely close (can happen with noisy lap start detection)
_dedup = [session_boundaries_unix[0]]
for _b in session_boundaries_unix[1:]:
    if abs(_b - _dedup[-1]) > 1e-3:
        _dedup.append(_b)
session_boundaries_unix = _dedup

# Segment durations (seconds) between session boundaries (unix seconds).
# boundaries: [win_start, lap_start1, lap_start2, ..., win_end]
session_segment_durations_s = [session_boundaries_unix[i + 1] - session_boundaries_unix[i]
                               for i in range(len(session_boundaries_unix) - 1)]

# "Full laps" within the session are only those between real lap boundaries inside the window:
# i.e. segments 1..(N-1), excluding:
#   segment 0: win_start -> first lap_start  (partial lap at session start)
#   last segment: last lap_start -> win_end  (partial lap at session end)
session_full_lap_durations_s = session_segment_durations_s[1:-1] if len(session_segment_durations_s) >= 3 else []
session_full_lap_end_unix = [session_boundaries_unix[i + 1] for i in range(1, len(session_boundaries_unix) - 2)] if len(
    session_boundaries_unix) >= 3 else []

# Kept for compatibility, but NOT used directly for display anymore.
SESSION_BEST_LAP_S = float(np.nanmin(session_full_lap_durations_s)) if session_full_lap_durations_s else float("nan")


def session_best_so_far(t_unix):
    """Best (minimum) full-lap duration completed so far within the session window.
    Returns 0.0 if no full laps have completed yet."""
    if not session_full_lap_durations_s:
        return 0.0
    # full lap k ends at session_full_lap_end_unix[k]
    best = float("inf")
    for dur, endu in zip(session_full_lap_durations_s, session_full_lap_end_unix):
        if endu <= t_unix and np.isfinite(dur):
            best = min(best, float(dur))
    return 0.0 if best == float("inf") else float(best)


def prev_full_lap(t_unix):
    """Previous completed full-lap duration within session. Returns 0.0 at session start."""
    if not session_full_lap_durations_s:
        return 0.0
    prev = 0.0
    for dur, endu in zip(session_full_lap_durations_s, session_full_lap_end_unix):
        if endu <= t_unix and np.isfinite(dur):
            prev = float(dur)
        else:
            break
    return prev


def _fmt_laptime(sec):
    if not np.isfinite(sec):
        return "--:--.---"
    sec = float(max(0.0, sec))
    m = int(sec // 60.0)
    s = sec - 60.0 * m
    return f"{m:d}:{s:06.3f}"


def lap_at_time(t_sample, t_out_s):
    """
    Returns (session_lap_time_s, prev_session_lap_s, lat_g, lon_g) for the current sample time.
    - Session is defined as the rendered/trimmed output window (output t=0 at vidStart).
    - Lap timer resets at the start of the rendered session (output t=0) and at each detected lap-start boundary
      from the lap CSV that occurs within the rendered window.
    """
    # t_sample is Haltech timebase (seconds) aligned to output video time via timeOff + sync
    # Convert to lap CSV timebase (unix seconds) using the precomputed offset.
    t_unix = t_sample + LAP_SYNC_OFFSET

    # Interpolate G's (these are just point samples; NaN if out of range)
    lat = float(np.interp(t_unix, lap_t, lap_lat, left=np.nan, right=np.nan))
    lon = float(np.interp(t_unix, lap_t, lap_lon, left=np.nan, right=np.nan))

    # Interpolate GPS (deg), if present
    if (lap_gps_lat is not None) and (lap_gps_lon is not None):
        gps_lat = float(np.interp(t_unix, lap_t, lap_gps_lat, left=np.nan, right=np.nan))
        gps_lon = float(np.interp(t_unix, lap_t, lap_gps_lon, left=np.nan, right=np.nan))
    else:
        gps_lat = np.nan
        gps_lon = np.nan

    # -------------------- Session-lap timing (session starts at output t=0) --------------------
    # Build boundaries from lap_starts that occur inside the rendered window.
    # Always include the rendered window start as boundary 0, so lap time is 0 at output start.
    # Note: win_start_unix / win_end_unix were computed from the rendered window.
    boundaries = session_boundaries_unix  # precomputed sorted list starting with win_start_unix and ending with win_end_unix

    # Find the index of the last boundary at or before t_unix
    # (t_unix is monotonic with output, but keep it robust)
    bi = int(np.searchsorted(boundaries, t_unix, side="right") - 1)
    if bi < 0:
        bi = 0
    if bi >= len(boundaries):
        bi = len(boundaries) - 1

    cur_boundary = float(boundaries[bi])
    session_lap_time = float(max(0.0, t_unix - cur_boundary))

    # Previous completed full-lap duration within session (session-only)
    prev_session_lap = float(prev_full_lap(t_unix))

    sess_best = float(session_best_so_far(t_unix))
    return session_lap_time, prev_session_lap, sess_best, lat, lon, gps_lat, gps_lon


# -------------------- Output (FFmpeg pipe; speed+size knobs) --------------------
# Uses VIDEO_OUT set at the top of the file.
FFMPEG_LOG = "ffmpeg_error.log"

# Choose encoder:
#   "h264_nvenc" = fastest/most compatible
#   "hevc_nvenc" = usually smaller at similar quality (still very fast on GPU)
USE_HEVC = False

# NVENC speed/size knobs (p1 fastest .. p7 highest quality)
NVENC_PRESET = "p2"  # try p1 or p2 for max speed
NVENC_CQ = "23"  # higher = smaller file / lower quality (typical 18-28)

# x264 fallback knobs
X264_PRESET = "veryfast"
X264_CRF = "23"  # higher = smaller file / lower quality (typical 20-28)

# NVENC extra “speed-first” knobs
NVENC_BFRAMES = "0"
NVENC_LOOKAHEAD = "0"
NVENC_AQ = "0"  # disables spatial/temporal AQ

# Scrub-friendly playback: force frequent keyframes (short GOP)
# Set to 1.0 for a keyframe about every second. Lower = easier scrubbing, larger files.


FFMPEG_FASTSTART = True  # moov atom at front (better for streaming)


def _start_ffmpeg(encoder: str):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH. Install it (e.g. winget install Gyan.FFmpeg).")

    try:
        open(FFMPEG_LOG, "wb").close()
    except Exception:
        pass
    log_f = open(FFMPEG_LOG, "ab")

    # --- Audio: sample-accurate trim via filter (fixes the "audio delayed" issue) ---
    # These are already computed earlier in your script:
    # AUDIO_TRIM_START_S = trim_start_s
    # AUDIO_TRIM_DUR_S   = None or (trim_end_s - trim_start_s)
    a_start = float(max(0.0, AUDIO_TRIM_START_S + AUDIO_TRIM_NUDGE_S))
    a_dur = AUDIO_TRIM_DUR_S if (AUDIO_TRIM_DUR_S is not None and AUDIO_TRIM_DUR_S > 0) else None

    # Build filter_complex only if we need to trim (you do in this script).
    use_audio_filter = (a_start > 0.0) or (a_dur is not None)

    cmd = [
        ffmpeg, "-y",
        "-hide_banner",
        "-loglevel", "error",

        # raw video in (we set exactly W/H/fps we are producing)
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}",
        "-r", f"{fps:.9f}",
        "-i", "pipe:0",

        # audio source (full file; we trim it in filter_complex for accuracy)
        "-i", VIDEO_IN,
    ]

    if use_audio_filter:
        # atrim is sample-accurate; asetpts resets timestamps; aresample async helps avoid drift
        if a_dur is None:
            af = f"[1:a]atrim=start={a_start:.6f},asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a]"
        else:
            af = f"[1:a]atrim=start={a_start:.6f}:duration={float(a_dur):.6f},asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[a]"

        cmd += [
            "-filter_complex", af,
            "-map", "0:v:0",
            "-map", "[a]",
            "-shortest",
        ]
    else:
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
        ]

    if TEST_RENDER_FIRST_10S:
        cmd += ["-t", str(TEST_DURATION_S)]

    # Keyframe interval for fast scrubbing
    try:
        _gop = int(max(1, round(float(fps) * float(SCRUB_KEYFRAME_EVERY_S))))
    except Exception:
        _gop = 30

    if encoder == "nvenc":
        if COMPRESS_OUTPUT:
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", COMPRESS_NVENC_PRESET,
                "-rc", "cbr",
                "-b:v", COMPRESS_VIDEO_BITRATE,
                "-maxrate", COMPRESS_VIDEO_BITRATE,
                "-bufsize", COMPRESS_VIDEO_BUFFER_SIZE,
                "-profile:v", "high",
                "-level:v", "5.1",
                "-tag:v", "avc1",
                "-g", str(_gop),
                "-forced-idr", "1",
            ]
        else:
            vcodec = "hevc_nvenc" if USE_HEVC else "h264_nvenc"
            cmd += [
                "-c:v", vcodec,
                "-preset", NVENC_PRESET,
                "-rc", "vbr",
                "-cq", str(NVENC_CQ),
                "-b:v", "0",
                "-bf", NVENC_BFRAMES,
                "-rc-lookahead", NVENC_LOOKAHEAD,
                "-tune", "ll",
                "-g", str(_gop),
                "-forced-idr", "1",
            ]
            if NVENC_AQ == "0":
                cmd += ["-spatial_aq", "0", "-temporal_aq", "0"]
            cmd += ["-profile:v", "high"]
    else:
        if COMPRESS_OUTPUT:
            cmd += [
                "-c:v", "libx264",
                "-preset", COMPRESS_X264_PRESET,
                "-b:v", COMPRESS_VIDEO_BITRATE,
                "-maxrate", COMPRESS_VIDEO_BITRATE,
                "-bufsize", COMPRESS_VIDEO_BUFFER_SIZE,
                "-profile:v", "high",
                "-level:v", "5.1",
                "-tag:v", "avc1",
                "-x264-params", f"keyint={_gop}:min-keyint={_gop}:scenecut=0:open-gop=0",
            ]
        else:
            cmd += ["-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-profile:v", "high",
                    "-x264-params", f"keyint={_gop}:min-keyint={_gop}:scenecut=0:open-gop=0"]

    # Video format
    cmd += ["-pix_fmt", "yuv420p"]

    # Audio: re-encode when filtered (required); otherwise you can still copy
    if use_audio_filter or COMPRESS_OUTPUT:
        audio_bitrate = COMPRESS_AUDIO_BITRATE if COMPRESS_OUTPUT else "192k"
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate]
    else:
        cmd += ["-c:a", "copy"]

    if FFMPEG_FASTSTART:
        cmd += ["-movflags", "+faststart"]

    cmd += [VIDEO_OUT]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f, bufsize=0)
    return p, cmd, log_f


def open_writer_with_fallback():
    p, cmd, log_f = _start_ffmpeg("nvenc")
    time.sleep(0.25)
    if p.poll() is not None:
        try:
            log_f.flush()
        except Exception:
            pass
        try:
            log_f.close()
        except Exception:
            pass
        print("\nNVENC failed (driver too old / unsupported). Falling back to libx264 CPU encoding.")
        print("See ffmpeg_error.log for details from the NVENC attempt.")
        p2, cmd2, log_f2 = _start_ffmpeg("x264")
        return p2, cmd2, log_f2, "x264"
    return p, cmd, log_f, ("hevc_nvenc" if USE_HEVC else "h264_nvenc")


def tempF(key, t_sample):
    raw = v(key, t_sample)
    k = (raw / SCALE["temp_k"]) if np.isfinite(raw) else np.nan
    return kelvin_to_f(k) if np.isfinite(k) else np.nan


# -------------------- FAST_RENDER: precompute frequently-used channels --------------------
# Removes repeated np.interp()/dict lookups inside the frame loop.
if FAST_RENDER:
    _t_out_arr = (np.arange(max_frames, dtype=np.float64) / float(fps))

    # Keys used directly in the render loop
    # Precompute per-frame sample times (log time for each output frame)
    _t_sample_arr = (MAIN_START_S + (np.arange(max_frames, dtype=np.float64) / float(fps))).astype(np.float64)

    _cache_keys = [
        "rpm", "tps", "speed", "map", "fuel_p", "fuel_p_exp",
        "wb1", "wb2", "lambda_tgt", "gear", "ign", "knock",
        "t_fl", "t_fr", "t_rl", "t_rr", "p_fl", "p_fr", "p_rl", "p_rr",
    ]

    for _k in _cache_keys:
        try:
            CHANNEL_CACHE[_k] = np.array([v_interp(_k, float(ts)) for ts in _t_sample_arr], dtype=np.float64)
        except Exception:
            pass

    # Brake percent is computed via brake_pct_at(); cache it too.
    try:
        CHANNEL_CACHE["__brake_pct"] = np.array([float(brake_pct_at(float(ts))) for ts in _t_sample_arr],
                                                dtype=np.float64)
    except Exception:
        pass

    # Rate-limited text cache (update at TEXT_UPDATE_HZ)
    _TEXT_EVERY_N = max(1, int(round(float(fps) / max(1e-6, float(TEXT_UPDATE_HZ)))))
    _lap_text_cache = {"lap": "", "prev": "", "sess": "", "day": "", "g_main": "", "g_lat": "", "g_lon": ""}
else:
    _TEXT_EVERY_N = 1
    _lap_text_cache = None

def render_video():
    global CURRENT_FRAME_I, _split_render_start_unix, _split_ready, _split_last_bi, _prog_hint_seg, _split_base_prog, _prev_lap_delta_s, _split_best_lap_s, _minimap_centerline_active, _dot_last_px, _dot_smooth_px, _dot_last_proj_m, _dot_last_prog, _dot_last_dir_px, _split_run_prev_p, _fused_prog, _fused_unix, _fused_locked, _fused_lock_t0, _fused_last_gps_p, _fused_gps_sane_count, _fused_centerline_ok_count, _minimap_rc_last_accept_p, _minimap_rc_last_accept_unix, _minimap_rc_disp_p, _prog_last, _prog_last_unix, _prog_last_xy_m, _prog_last_accept, _prog_pending_p, _prog_pending_xy_m, _prog_pending_count, cap

    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_IN)
        if trim_start_s > 0.0:
            start_frame = int(round(trim_start_s * fps))
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                cap.set(cv2.CAP_PROP_POS_MSEC, trim_start_s * 1000.0)

    ffmpeg_proc, ffmpeg_cmd, ffmpeg_log_f, encoder_used = open_writer_with_fallback()
    print(f"Encoder: {encoder_used}")
    print(f"Output: {W}x{H} @ {fps:.3f} fps (matches source as-probed/OpenCV)")

    start_time = time.time()
    last_report_time = start_time
    last_report_frame = 0

    panel_w, panel_h = S(680), S(300)
    bar_w = S(28)
    gap = S(14)

    panel_x = W - S(20) - panel_w - bar_w - gap
    panel_y = H - S(20) - panel_h
    brake_x = panel_x - gap - bar_w
    throttle_x = panel_x + panel_w + gap

    rpmbar_x = brake_x
    rpmbar_w = (throttle_x + bar_w) - brake_x
    rpmbar_y = panel_y - S(70)

    tires_w = S(76 * 2 + 22 + 10)
    # tire panel is a 2x2 grid of square icons; height matches width

    tires_h = S(120 * 2 + 18)  # height for 2 rows of tires + gap; prevents RL/RR clipping

    i = 0
    print("Starting render loop...")
    # Decode frames via ffmpeg pipe (much faster + reliable seeking vs OpenCV on Windows)
    ffdec_proc, ff_read_frame = start_ffmpeg_frame_reader(VIDEO_IN, W, H, fps, trim_start_s)

    _first_frame_written = False
    while True:
        if i >= max_frames:
            break

        frame = ff_read_frame()
        if frame is None:
            break

            print(f"...working (frame {i})", flush=True)

        # Output time starts at 0 at VIDEO_TRIM_START_S
        t_out = i / fps
        CURRENT_FRAME_I = i
        t_src = trim_start_s + t_out

        # Main log sampling uses OUTPUT time base (0 at trimmed start)
        t_sample = MAIN_START_S + t_out

        rpm = v("rpm", t_sample)

        tps_raw = v("tps", t_sample)
        tps = (tps_raw / SCALE["tps"]) if np.isfinite(tps_raw) else 0.0
        tps = float(np.clip(tps, 0.0, 100.0))

        brake = (CHANNEL_CACHE.get('__brake_pct')[i] if (FAST_RENDER and '__brake_pct' in CHANNEL_CACHE) else brake_pct_at(
            t_sample))

        speed_raw = v("speed", t_sample)
        speed_kph = (speed_raw / SCALE["speed"]) if np.isfinite(speed_raw) else np.nan
        mph = (speed_kph * KPH_TO_MPH) if np.isfinite(speed_kph) else np.nan

        map_raw = v("map", t_sample)
        map_kpa_abs = (map_raw / SCALE["kpa"]) if np.isfinite(map_raw) else np.nan
        map_psi_abs = kpa_to_psi(map_kpa_abs) if np.isfinite(map_kpa_abs) else np.nan
        map_psig = psi_abs_to_psig(map_psi_abs) if np.isfinite(map_psi_abs) else np.nan

        fp_raw = v("fuel_p", t_sample)
        fpe_raw = v("fuel_p_exp", t_sample)
        fp_kpa_abs = (fp_raw / SCALE["fuel_kpa"]) if np.isfinite(fp_raw) else np.nan
        fpe_kpa_abs = (fpe_raw / SCALE["fuel_kpa_exp"]) if np.isfinite(fpe_raw) else np.nan
        fp_psi_abs = kpa_to_psi(fp_kpa_abs) if np.isfinite(fp_kpa_abs) else np.nan
        fpe_psi_abs = kpa_to_psi(fpe_kpa_abs) if np.isfinite(fpe_kpa_abs) else np.nan
        fp_psig = psi_abs_to_psig(fp_psi_abs) if np.isfinite(fp_psi_abs) else np.nan
        fpe_psig = psi_abs_to_psig(fpe_psi_abs) if np.isfinite(fpe_psi_abs) else np.nan
        if np.isfinite(fp_psig): fp_psig = max(0.0, fp_psig)
        if np.isfinite(fpe_psig): fpe_psig = max(0.0, fpe_psig)

        if np.isfinite(fp_psig) and np.isfinite(fpe_psig) and abs(fpe_psig) > 1e-6:
            diff_pct = (fp_psig - fpe_psig) / fpe_psig * 100.0
            fuel_pct = f"{diff_pct:+0.1f}%"
        else:
            fuel_pct = ""

        clt = tempF("clt", t_sample)
        iat = tempF("iat", t_sample)
        oil = tempF("oil_temp", t_sample)

        wb1 = v("wb1", t_sample)
        wb2 = v("wb2", t_sample)
        lt = v("lambda_tgt", t_sample)
        wb1 = (wb1 / SCALE["lambda"]) if np.isfinite(wb1) else np.nan
        wb2 = (wb2 / SCALE["lambda"]) if np.isfinite(wb2) else np.nan
        lt = (lt / SCALE["lambda"]) if np.isfinite(lt) else np.nan

        gear_raw = v("gear", t_sample)

        ign_raw = v("ign", t_sample)
        knock_raw = v("knock", t_sample)
        ign = (ign_raw / IGN_SCALE) if np.isfinite(ign_raw) else np.nan
        knock = (knock_raw / KNOCK_SCALE) if np.isfinite(knock_raw) else np.nan

        data = {
            "rpm": f"{rpm:0.0f}" if np.isfinite(rpm) else "--",
            "gear": f"{int(gear_raw)}" if np.isfinite(gear_raw) else "--",
            "mph": f"{mph:0.1f}" if np.isfinite(mph) else "--",
            "map": f"{map_psig:0.1f} psi" if np.isfinite(map_psig) else "--",
            "clt": f"{clt:0.0f}F" if np.isfinite(clt) else "--",
            "oil": f"{oil:0.0f}F" if np.isfinite(oil) else "--",
            "iat": f"{iat:0.0f}F" if np.isfinite(iat) else "--",
            "tps": f"{tps:0.0f}%",
            "fuel_psi": f"{fp_psig:0.1f} psi" if np.isfinite(fp_psig) else "--",
            "fuel_pct": fuel_pct,
            "wb": f"{wb1:0.2f}/{wb2:0.2f}" if np.isfinite(wb1) and np.isfinite(wb2) else "--",
            "lt": f"{lt:0.2f}" if np.isfinite(lt) else "--",
            "ign": f"Ign {ign:0.1f}" if np.isfinite(ign) else "Ign --",
            "knock": f"Knock {int(round(knock))}" if np.isfinite(knock) else "Knock --",
        }

        vals = {
            "t_fl": v("t_fl", t_sample), "t_fr": v("t_fr", t_sample),
            "t_rl": v("t_rl", t_sample), "t_rr": v("t_rr", t_sample),
            "p_fl": v("p_fl", t_sample), "p_fr": v("p_fr", t_sample),
            "p_rl": v("p_rl", t_sample), "p_rr": v("p_rr", t_sample),
        }

        draw_panel_roi(frame, panel_x, panel_y, panel_w, panel_h, draw_bottom_right, data, 0, 0, panel_w, panel_h)
        draw_bar(frame, brake, brake_x, panel_y, bar_w, panel_h, (0, 0, 255))
        draw_bar(frame, tps, throttle_x, panel_y, bar_w, panel_h, (0, 255, 0))
        draw_rpm_bar(frame, rpm, rpmbar_x, rpmbar_y, rpmbar_w, S(54))
        draw_panel_roi(frame, W - tires_w - S(20), S(20), tires_w, tires_h, draw_tires, vals, 0, 0, SCALE["temp_k"],
                       SCALE["tire_kpa"])
        lap_time_s, prev_lap_s, sess_best_s, lat_g, lon_g, gps_lat_deg, gps_lon_deg = lap_at_time(t_sample, t_out)
        # ----- Split computation (delta vs current session-best lap at same track progress) -----
        t_unix_now = t_sample + LAP_SYNC_OFFSET
        speed_mps_now = (mph * 0.44704) if np.isfinite(mph) else None
        if _split_render_start_unix is None:
            # Only show splits after we have a FULL completed lap that starts after the rendered segment begins.
            _split_render_start_unix = float(t_unix_now)
            _split_ready = False
        # determine current session segment index
        bi_now = int(np.searchsorted(session_boundaries_unix, t_unix_now, side='right') - 1)
        if bi_now < 0:
            bi_now = 0
        if bi_now >= len(session_boundaries_unix):
            bi_now = len(session_boundaries_unix) - 1
        # elapsed time since start of this session/lap segment (stable even if video starts mid-lap)
        seg_time_s = float(max(0.0, t_unix_now - float(session_boundaries_unix[bi_now])))
        if _split_last_bi is None:
            _split_last_bi = bi_now
            _prog_hint_seg = None
            _split_base_prog = _progress_from_gps(gps_lat_deg, gps_lon_deg)
            _reset_virtual_splits()
        # if we crossed into a new segment, the previous one is now complete (except the final win_end segment)
        if bi_now != _split_last_bi:
            ended = _split_last_bi
            # ended segment is [boundary[ended], boundary[ended+1]] if it exists
            if 0 <= ended < (len(session_boundaries_unix) - 2):
                t0 = float(session_boundaries_unix[ended])
                t1 = float(session_boundaries_unix[ended + 1])
                dur = t1 - t0
                # Only accept a reference lap if it is a *FULL LAP segment* inside the rendered window.
                # Full laps are segments 1..(len(boundaries)-3). Segment 0 is the initial partial lap (win_start -> first lap boundary),
                # and the last segment is the final partial lap (last lap boundary -> win_end).
                if (ended >= 1) and (ended < (len(session_boundaries_unix) - 2)):
                    t0 = float(session_boundaries_unix[ended])
                    t1 = float(session_boundaries_unix[ended + 1])
                    dur = t1 - t0
                    # update ref only if this completed full lap is the new session best
                    best_before = float(_split_best_lap_s)
                    # Store last-lap delta vs the session best *before* updating (so improvements show as negative).
                    if np.isfinite(best_before):
                        _prev_lap_delta_s = float(dur) - best_before
                    else:
                        _prev_lap_delta_s = 0.0

                    # If this completed full lap is the new session best, update the reference
                    if dur > 20.0 and dur < 600.0 and dur < _split_best_lap_s:
                        _split_best_lap_s = float(dur)
                        _make_split_reference(t0, t1)
                        _split_ready = (_split_ref_prog is not None) and (_split_ref_time is not None)
            _split_last_bi = bi_now
            _prog_hint_seg = None
            _split_base_prog = _progress_from_gps(gps_lat_deg, gps_lon_deg)
            _reset_virtual_splits()
        # compute current split (virtual splits by default)
        split_s = np.nan
        split_idx = 0
        rel_p = np.nan
        if _split_ready and (_split_ref_prog is not None) and (_split_ref_time is not None) and np.isfinite(lap_time_s):
            raw_p = _progress_from_gps(gps_lat_deg, gps_lon_deg)
            if np.isfinite(raw_p) and np.isfinite(_split_base_prog):
                rel_p = raw_p - float(_split_base_prog)
                if rel_p < 0.0:
                    rel_p += 1.0
                # clamp into [0,1]
                rel_p = float(max(0.0, min(1.0, rel_p)))
                # Smooth/monotonic progress for *running* split display (prevents oscillation after lap 1)
                rel_p_use = _filter_running_rel_p(rel_p, speed_mps_now, 1.0 / float(fps if fps else 30.0)) if VIRTUAL_SPLITS_RUNNING else rel_p
                # Early-lap sanity: if projection is still "stuck" near some later part of the lap right after SF,
                # the running split can show huge negatives (e.g. -25s). Detect and reset the split base in that case.
                if lap_time_s < 6.0 and rel_p_use > 0.15:
                    # Treat current raw progress as the new lap start reference.
                    _split_base_prog = float(raw_p)
                    rel_p = 0.0
                    rel_p_use = 0.0
                    _split_run_prev_p = np.nan
                ref_t_now = float(np.interp(rel_p_use, _split_ref_prog, _split_ref_time))
                # Running delta vs session-best at current progress (stable; avoids impossible huge +/- values)
                # Use lap_time_s (elapsed since lap start) vs reference lap elapsed at rel_p_use.
                split_s = float(lap_time_s - ref_t_now)
                # Sanity: if split delta is wildly large early in a lap, rel_p_use likely wrapped to the wrong
                # part of the centerline. Rebase the running-split progress so we don't show impossible values
                # like -25s right after the start/finish.
                if np.isfinite(split_s) and (abs(split_s) > SPLIT_SANITY_MAX_ABS_S) and (lap_time_s < SPLIT_SANITY_MAX_LAP_T_S):
                    _split_base_prog = float(raw_p)
                    rel_p = 0.0
                    rel_p_use = 0.0
                    _split_run_prev_p = np.nan
                    ref_t_now = 0.0
                    split_s = float(lap_time_s)
                if VIRTUAL_SPLITS_ENABLE and (VIRTUAL_SPLITS_N is not None) and int(VIRTUAL_SPLITS_N) > 1:
                    N = int(VIRTUAL_SPLITS_N)
                    if VIRTUAL_SPLITS_RUNNING:
                        # Show which split/sector we're currently IN (1..N)
                        split_idx = int(min(N - 1, max(0, int(rel_p_use * N)))) + 1
                    else:
                        # Old behavior: show last passed split only
                        split_idx, split_s = _update_virtual_splits(rel_p, float(seg_time_s))
        # Update expensive lap/G text at a lower rate (looks identical, much faster)
        if FAST_RENDER and (_TEXT_EVERY_N > 1) and (i % _TEXT_EVERY_N == 0):
            _lap_text_cache["lap"] = f"Lap  {_fmt_laptime(lap_time_s)}"
            _lap_text_cache["prev"] = f"Prev {_fmt_laptime(prev_lap_s)}"
            # Split display: hide until we have a completed reference lap (no splits on first lap)
            if not _split_ready:
                _lap_text_cache["split"] = "Split --.--"
            else:
                if VIRTUAL_SPLITS_ENABLE and (VIRTUAL_SPLITS_N is not None) and int(VIRTUAL_SPLITS_N) > 1:
                    N = int(VIRTUAL_SPLITS_N)
                    # Running split: show current sector index (1..N) and delta at current progress
                    if np.isfinite(split_s):
                        split_sign = "-" if split_s < 0 else "+" if split_s > 0 else " "
                        _lap_text_cache["split"] = f"S{int(split_idx)}/{N}  {split_sign}{abs(split_s):0.2f}"
                    else:
                        _lap_text_cache["split"] = f"S{int(split_idx)}/{N}  --.--"
                else:
                    if np.isfinite(split_s):
                        split_sign = "-" if split_s < 0 else "+" if split_s > 0 else " "
                        _lap_text_cache["split"] = f"Split {split_sign}{abs(split_s):0.2f}"
                    else:
                        _lap_text_cache["split"] = "Split --.--"
            _lap_text_cache["sess"] = f"Sess best {_fmt_laptime(sess_best_s)}"
            _lap_text_cache["day"] = f"Day best  {_fmt_laptime(DAY_BEST_LAP_S)}"
            if np.isfinite(lat_g) and np.isfinite(lon_g):
                g_mag = math.sqrt(float(lat_g) * float(lat_g) + float(lon_g) * float(lon_g))
                _lap_text_cache["g_main"] = f"{g_mag:0.2f}g"
                _lap_text_cache["g_lat"] = f"Lat {float(lat_g):+0.2f}g"
                _lap_text_cache["g_lon"] = f"Lon {float(lon_g):+0.2f}g"
            else:
                _lap_text_cache["g_main"] = "--.--g"
                _lap_text_cache["g_lat"] = "Lat --.--g"
                _lap_text_cache["g_lon"] = "Lon --.--g"

        draw_lap_panel(frame, lap_time_s, prev_lap_s, sess_best_s, split_s, lat_g, lon_g,
                       txt_cache=_lap_text_cache if FAST_RENDER else None)
        draw_minimap(frame, gps_lat_deg, gps_lon_deg, t_unix=t_sample, speed_mps=speed_mps_now)

        # Write frame to ffmpeg stdin (must be contiguous bytes)
        frame_bytes = memoryview(np.ascontiguousarray(frame)).cast("B")
        try:
            ffmpeg_proc.stdin.write(frame_bytes)
            if not _first_frame_written:
                _first_frame_written = True
                print("First frame written to encoder.", flush=True)
        except (BrokenPipeError, OSError) as e:
            rc = ffmpeg_proc.poll()
            raise RuntimeError(f"ffmpeg write failed: {e} (returncode={rc}). See {FFMPEG_LOG}. Command: {ffmpeg_cmd}")

        # progress once per second of video
        if max_frames > 0 and (i % int(max(1, fps)) == 0):
            now = time.time()
            elapsed = now - start_time
            progress = (i + 1) / max_frames
            eta = (elapsed / progress) - elapsed if progress > 0 else 0.0
            dt = now - last_report_time
            frame_delta = (i + 1) - last_report_frame
            proc_fps = (frame_delta / dt) if dt > 1e-6 else 0.0
            print(
                f"\rProcessing: {progress * 100:5.1f}% | Frame {i + 1}/{max_frames} | Proc {proc_fps:5.1f} fps | ETA {eta / 60:5.1f} min",
                end="", flush=True)
            last_report_time = now
            last_report_frame = (i + 1)

        i += 1

    cap.release()

    # Close ffmpeg stdin and wait for encode to finish
    try:
        ffmpeg_proc.stdin.close()
    except Exception:
        pass
    rc = ffmpeg_proc.wait()
    try:
        ffmpeg_log_f.close()
    except Exception:
        pass

    if rc != 0:
        print("\nffmpeg failed. See ffmpeg_error.log for details.")
        raise SystemExit(1)

    elapsed_total = time.time() - start_time
    print("\nRender complete (with audio).")
    print("Output written to:", VIDEO_OUT)
    print(f"Total time: {elapsed_total / 60:.2f} min")

if __name__ == "__main__":
    render_video()
