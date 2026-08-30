# Track Overlay Analytics

Track Overlay Analytics renders synchronized vehicle telemetry onto track-day video. It combines a source video, its matching telemetry log, and a RaceChrono session export to produce a new video with driving data, lap information, virtual splits, and a GPS minimap.

The repository includes a 100-second 4K sample session (`GX010055`) so the current configuration can be run without sourcing additional data.

## Overlay features

- Speed, RPM, gear, ignition timing, and knock data
- Fuel pressure and level
- Tire pressure and temperature visualization
- Lateral and longitudinal G-force
- Current, previous, and session-best lap timing
- Configurable running virtual splits
- GPS track map with a smoothed position marker
- Source-resolution rendering with synchronized audio
- NVIDIA NVENC encoding when available, with an automatic `libx264` fallback

## Requirements

- Python 3.11 or newer
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` available on `PATH`
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

## Included sample

| File | Purpose |
| --- | --- |
| `GX010055.mp4` | Source track video |
| `GX010055.csv` | Main vehicle telemetry log |
| `session_20251220_093244_grange_v3.csv` | RaceChrono GPS and lap-session export |
| `dataMarks.txt` | Trim and synchronization anchors |
| `newMain.py` | Overlay renderer and configuration |

## Run the sample

Run the script from the repository root because its input paths are relative to the working directory:

```bash
python newMain.py
```

The configured sample writes `GX010055Final.mp4`. Rendered `*Final.mp4` files and FFmpeg error logs are ignored by Git.

The default compressed-output profile preserves the source resolution while targeting a 7 Mbit/s H.264 video stream and 128 kbit/s AAC audio. Set `COMPRESS_OUTPUT = False` in the user configuration section if that profile is not desired.

## Configure another session

The primary settings are grouped under `USER CONFIG` near the top of `newMain.py`:

1. Add the session base name to `NUMS` and select it with `NUM`.
2. Set `LAP_CSV_IN` to the corresponding RaceChrono export.
3. Add a synchronization row to `dataMarks.txt`.
4. Adjust rendering, minimap, virtual-split, temperature, or UI-scale options as needed.

Each non-comment row in `dataMarks.txt` uses this pipe-delimited format:

```text
base_name|trim_start|trim_end|video_sync_anchor|log_sync_anchor|fine_tune_ms
```

Times may use `m:ss`, `m:ss.mmm`, or plain seconds where supported. The sample row is:

```text
GX010055|0:00|1:40|0:20.400|4:04.560|900
```

The video and main telemetry CSV are expected to share the selected base name, such as `GX010055.mp4` and `GX010055.csv`. Keep the synchronization anchors tied to a clearly identifiable event visible in the video and present in the log.

## Faster test renders

Before a full render, set `TEST_RENDER_FIRST_10S = True` and choose `TEST_DURATION_S` in `newMain.py`. This limits the output duration while exercising the same decoding, overlay, audio, and encoding pipeline.

## License

This project is available under the [MIT License](LICENSE).
