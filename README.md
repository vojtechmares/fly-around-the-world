# Fly around the world

## Travel plan

- Nov 14: Prague (CZ) -> Frankfurt (DE) (transfer only)
- Nov 14: Frankfurt (DE) -> Seoul (KR)
- TBD: Seoul (KR) -> TBD(Japan): Tokyo / Osaka
- Nov 24: Tokyo (JP) -> San Francisco (US) (transfer only)
- Nov 24: San Francisco (US) -> Las Vegas (US)
- Dec 4: Las Vegas (US) -> New York City (US)
- Dec 14: New York City (US) -> London (UK) (transfer only)
- Dec 15: London (UK) -> Prague (CZ)

## Globe animation

### Setup

Requires macOS on Apple Silicon (Metal GPU, VideoToolbox encoder, system fonts Avenir Next and DIN Alternate), [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`).

```sh
uv sync
uv run python prep.py   # downloads the NASA textures (~110 MB) into assets/ and builds assets/cache/ (~1.2 GB)
```

`prep.py` verifies the downloads by checksum, resumes interrupted downloads and skips files that are already there, so it is safe to run again. `assets/` and `out/` are gitignored.

### Usage

`render.py` renders a 3D globe animation of the trip to MP4. What the video does is described by a script in `scripts/`:

- `trip.toml` - stops and legs, shared by the scripts via `include`
- `around-the-world.toml` - full version: title intro, a stop after every leg, outro over the North Pole with trip stats
- `leaps.toml` - no intro and no outro, connections flown through without stopping, 2 s pause at each destination

A script can override anything from the files it includes. The `[timing]` keys and their defaults are listed in `TIMING` in `render.py`. Change a stop (e.g. Tokyo -> Osaka) in the `legs` of `trip.toml`; `OSA` is already defined. A leg with `transfer = true` marks its destination as a connection only.

```sh
uv run python render.py scripts/leaps.toml --preset 1080 --stills 5,27  # preview frames -> out/stills/
uv run python render.py scripts/leaps.toml --preset 4k                  # -> out/leaps-4k.mp4
uv run python render.py scripts/leaps.toml --preset 4k --vertical       # 9:16 for Reels/Shorts/TikTok
```

The globe, stars, atmosphere and final compositing run on the GPU (`globe.wgsl`, driven by `gpu.py` through wgpu/Metal). Routes, markers, labels and the HUD are drawn with Pillow in worker processes, and video is encoded by the hardware H.264 encoder (VideoToolbox). A 4K video of about a minute renders in about 2 minutes.

Options: `--fps` (default 30), `--workers` (overlay processes, default 3), `--start/--end` (seconds), `--encoder vt|x264` (default `vt`, hardware), `--quality` (VideoToolbox, 1-100, default 75), `--crf` (x264, default 15), `--out`.

Videos are rendered in segments under `out/parts/`, so an interrupted render resumes where it stopped. Segments are reused only while the script, the code and the encoder settings are unchanged.

A 1080p file from a 4K master (sharper than rendering 1080p directly):

```sh
ffmpeg -i out/leaps-4k.mp4 -vf scale=1920:1080:flags=lanczos -c:v h264_videotoolbox -q:v 75 -profile:v high \
  -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart out/leaps-1080p.mp4
```

Textures: NASA Visible Earth - Blue Marble Next Generation (November 2004, topography and bathymetry), Black Marble 2016 night lights, cloud composite and GEBCO bathymetry (water mask). Public domain, courtesy of NASA.
