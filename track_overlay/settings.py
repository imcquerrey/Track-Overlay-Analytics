"""User-editable renderer settings shared by the entry point and helpers."""

# ============================================================
# USER CONFIG (things you actually change)
# ============================================================

# --- Input / Output ---
NUMS = ["GX010055", "GX010056"]

NUM = NUMS[0]  # base name for your per-session files (e.g. 2.mp4 + 2.csv)

VIDEO_IN = f"{NUM}.mp4"  # input video
CSV_IN = f"{NUM}.csv"  # main telemetry CSV used by the overlay
# RaceChrono export that contains GPS + lap info
LAP_CSV_IN = "session_20251220_093244_grange_v3.csv"
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
# max dot movement per frame in minimap pixels (prevents teleports)
MINIMAP_DOT_MAX_PX_STEP = 8.0
MINIMAP_DOT_MIN_PX_STEP = 0.0  # set to ~0.8 if you want to avoid subpixel 'stalls'
MINIMAP_DOT_SMOOTH_ALPHA = 0.25
MINIMAP_DOT_STALL_EPS_PX = 0.15

MINIMAP_ENABLE = True
MINIMAP_POS = "topleft"  # topleft / topright / bottomleft / bottomright

# Minimap dot stability (prevents rare "teleport" snaps at hairpins / S-F)
# Smaller SEARCH_WINDOW_SEGS reduces the chance of snapping to a nearby parallel segment.
# segments to search around last known segment (circular)
MINIMAP_DOT_SEARCH_WINDOW_SEGS = 90
MINIMAP_DOT_MAX_JUMP_M = 25.0  # hard gate: max meters the dot may move in one frame
# gate on along-track progress step (0..1), cyclic (higher = more permissive)
MINIMAP_DOT_MAX_STEP_PROG = 0.06
# hard cap on cross-track jump (m) to prevent parallel-segment flips
MINIMAP_MAX_LATERAL_JUMP_M = 1.5
MINIMAP_CONFIRM_STREAK = 6  # frames required to accept a large lateral move
MINIMAP_CONFIRM_LATERAL_M = 1.5  # lateral threshold (m) for debounce
MINIMAP_CONFIRM_JUMP_M = 6.0  # total jump threshold (m) for debounce

# --- Virtual splits (recommended) ---
VIRTUAL_SPLITS_ENABLE = True
VIRTUAL_SPLITS_N = 6  # 3 / 4 / 6 / 8 etc.
VIRTUAL_SPLITS_RUNNING = True  # True: show a running delta inside the current sector
# Running-split sanity clamp: if delta goes insane early in lap, it's almost always a progress wrap/glitch.
# if |split| exceeds this, rebase progress for running split
SPLIT_SANITY_MAX_ABS_S = 10.0
# only apply rebasing within first N seconds of a lap
SPLIT_SANITY_MAX_LAP_T_S = 25.0

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

FAST_RENDER = True
TEXT_UPDATE_HZ = 10.0
FAST_NO_AA = True
KPA_TO_PSI = 0.1450377377
KPH_TO_MPH = 0.621371
ATM_PSI = 14.6959
