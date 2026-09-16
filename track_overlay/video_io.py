"""Video metadata probing and raw-frame decoding."""

import json
import shutil
import subprocess

import numpy as np


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
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. winget install Gyan.FFmpeg).")

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
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=10 ** 7)
    frame_bytes = int(W) * int(H) * 3

    def read_frame():
        buf = p.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape((H, W, 3)).copy()

    return p, read_frame
