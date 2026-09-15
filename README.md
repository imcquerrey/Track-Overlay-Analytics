# Track Overlay Analytics

Track Overlay Analytics renders synchronized vehicle telemetry onto track-day video. It combines source video, matching vehicle telemetry, and a RaceChrono session export to produce a video with driving data, lap information, virtual splits, and a GPS minimap.

This repository includes two named 100-second 4K sample sessions so rendering and session-isolation work can be developed entirely from local data. `GX010055` and `GX010056` intentionally contain the same source media and telemetry under different session names; this makes it possible to compare a session rendered alone with the same session rendered after another one.

## Overlay features

- Speed, RPM, gear, ignition timing, and knock data
- Fuel pressure and level
- Tire pressure and temperature visualization
- Lateral and longitudinal G-force
- Current, previous, and session-best lap timing
- Configurable running virtual splits
- GPS track map with a smoothed position marker
- Source-resolution rendering with synchronized audio
- NVIDIA NVENC encoding with a `libx264` fallback

## Requirements

- Python 3.11 or newer
- FFmpeg and `ffprobe` available on `PATH`
- The Python packages pinned in `requirements.txt`

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

FFmpeg is a separate system dependency and is not installed by `pip`. Verify both executables before rendering:

```bash
ffmpeg -version
ffprobe -version
```

## Included sample data

| File | Purpose |
| --- | --- |
| `GX010055.mp4` | First source track video |
| `GX010055.csv` | First vehicle telemetry log |
| `GX010056.mp4` | Second named source-video fixture |
| `GX010056.csv` | Second named telemetry fixture |
| `session_20251220_093244_grange_v3.csv` | RaceChrono GPS and lap-session export shared by the fixtures |
| `dataMarks.txt` | Per-session trim and synchronization anchors |
| `newMain.py` | Overlay renderer and user configuration |

The media files are large because they retain the 3840x2160 source resolution and audio needed to exercise the real decode, overlay, and encode pipeline.

## Current baseline behavior

The `NUMS` configuration contains both sample names, but the current implementation assigns `NUM = NUMS[0]` and renders only the first session. This repository is therefore a reproducible baseline for adding correct multi-session processing and isolating mutable rendering state between sessions.

Run the current baseline from the repository root because its input paths are relative to the working directory:

```bash
python newMain.py
```

It writes `GX010055Final.mp4`. Rendered `*Final.mp4` files and FFmpeg error logs are ignored by Git.

The default compressed-output profile preserves the source resolution while targeting a 7 Mbit/s H.264 video stream and 128 kbit/s AAC audio. Set `COMPRESS_OUTPUT = False` in the user configuration section to use the alternative quality-oriented encoder configuration.

## Session configuration

Each session name expects a video and main telemetry CSV with the same base name, such as `GX010055.mp4` and `GX010055.csv`. Each non-comment row in `dataMarks.txt` uses this pipe-delimited format:

```text
base_name|trim_start|trim_end|video_sync_anchor|log_sync_anchor|fine_tune_ms
```

The included rows are:

```text
GX010055|0:00|1:40|0:20.400|4:04.560|900
GX010056|0:00|1:40|0:20.400|4:04.560|900
```

Keep synchronization anchors tied to a clearly identifiable event visible in the video and present in the telemetry log. The shared RaceChrono filename and the rendering, minimap, virtual-split, temperature, compression, and UI-scale options are configured near the top of `newMain.py`.

## Short test renders

Before a full render, set `TEST_RENDER_FIRST_10S = True` and reduce `TEST_DURATION_S` in `newMain.py`. This limits output duration while exercising the same decoding, overlay, audio, and encoding pipeline. Generated media should not be committed.

## License

This project is available under the [MIT License](LICENSE).
