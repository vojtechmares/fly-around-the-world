# Fly around the world

A 3D animation of flying around the globe from one trip stop to the next, rendered to MP4 for vlogs and social networks. It renders in 4K, landscape or vertical 9:16, and is driven by editable TOML scripts.

## Travel plan

- Nov 14: Prague (CZ) -> Frankfurt (DE) (transfer only)
- Nov 14: Frankfurt (DE) -> Seoul (KR)
- TBD: Seoul (KR) -> TBD(Japan): Tokyo / Osaka
- Nov 24: Tokyo (JP) -> San Francisco (US) (transfer only)
- Nov 24: San Francisco (US) -> Las Vegas (US)
- Dec 4: Las Vegas (US) -> New York City (US)
- Dec 14: New York City (US) -> London (UK) (transfer only)
- Dec 15: London (UK) -> Prague (CZ)

The plan is encoded in [`scripts/trip.toml`](scripts/trip.toml).

## Requirements

- macOS on Apple Silicon (Metal GPU, VideoToolbox hardware encoder, system fonts Avenir Next and DIN Alternate)
- [uv](https://docs.astral.sh/uv/) and ffmpeg: `brew install uv ffmpeg`

## Setup

```sh
uv sync
uv run python prep.py
```

`prep.py` downloads the NASA textures (~110 MB) into `assets/` and builds the texture cache in `assets/cache/` (~1.2 GB). It verifies the downloads by checksum, resumes interrupted downloads and skips files that are already there, so it is safe to run again. `assets/` and `out/` are gitignored.

## Rendering

```sh
uv run python render.py scripts/leaps.toml --preset 1080 --stills 5,27  # preview frames -> out/stills/
uv run python render.py scripts/leaps.toml --preset 4k                  # -> out/leaps-4k.mp4
uv run python render.py scripts/leaps.toml --preset 4k --vertical       # -> out/leaps-4k-vertical.mp4
```

A video of about a minute renders in about 2 minutes in 4K. The output is named after the script: `out/<script>-<preset>[-vertical].mp4`.

| Option | Default | Description |
| --- | --- | --- |
| `--preset` | `preview` | `preview` (960x540), `1080` (1920x1080) or `4k` (3840x2160) |
| `--vertical` | off | 9:16 frame for Reels, Shorts and TikTok |
| `--stills` | | Comma separated times (s) to render as PNG instead of a video |
| `--start`, `--end` | whole video | Time range to render (s) |
| `--fps` | 30 | Frame rate |
| `--encoder` | `vt` | `vt` (hardware, VideoToolbox) or `x264` (software) |
| `--quality` | 75 | VideoToolbox quality, 1-100 |
| `--crf` | 15 | x264 quality, lower is better |
| `--workers` | 3 | Processes drawing the 2D overlays |
| `--out` | | Output file |

Videos are rendered in segments under `out/parts/`, so an interrupted render resumes where it stopped. Segments are reused only while the script, the code and the encoder settings are unchanged.

A 1080p file made from a 4K master is sharper than one rendered in 1080p directly:

```sh
ffmpeg -i out/leaps-4k.mp4 -vf scale=1920:1080:flags=lanczos -c:v h264_videotoolbox -q:v 75 -profile:v high \
  -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart out/leaps-1080p.mp4
```

## Scripts

A script in `scripts/` describes one video:

- [`trip.toml`](scripts/trip.toml) - title, stops and legs, shared by the scripts via `include`
- [`around-the-world.toml`](scripts/around-the-world.toml) - title intro, a stop after every leg, outro over the North Pole with trip stats
- [`leaps.toml`](scripts/leaps.toml) - no intro and no outro, connections flown through without stopping, 2 s pause at each destination

A script can `include` other files and override anything from them. To make a new version, copy a script and change its `[timing]`.

Stops are defined by a code, name, country, coordinates and the side of the label. Legs are flown in order. A leg with `transfer = true` marks its destination as a connection only, and `date` is shown in the HUD. To change a stop (e.g. Tokyo -> Osaka), edit the `legs` in `trip.toml`; `OSA` is already defined.

| `[timing]` key | Default | Description |
| --- | --- | --- |
| `intro` | `true` | Title card and zoom in from space |
| `intro_duration` | 5.5 | Seconds |
| `outro` | `true` | Rise over the North Pole and show the stats |
| `outro_duration` | 11.5 | Seconds |
| `start_hold` | 1.4 | Seconds on the first city before take-off |
| `hold` | 2.0 | Seconds at each destination |
| `transfer_hold` | 0.7 | Seconds at connections, unless flying through them |
| `final_hold` | 1.2 | Seconds at the last city (before the outro, if any) |
| `fly_through_transfers` | `false` | Continue from a connection without stopping |
| `flight_speed` | 1.0 | Multiplier of the flight speed |
| `fade_in` | 0.6 | Seconds of fade from black, 0 for none |
| `fade_out` | 1.0 | Seconds of fade to black, 0 for none |

## How it works

- [`render.py`](render.py) - scripts, timeline and camera, the 2D overlays (routes, plane, labels, HUD, title and end cards) drawn with Pillow in worker processes, and the encoding
- [`gpu.py`](gpu.py) - GPU renderer (wgpu on Metal): uploads the textures and composites each frame
- [`globe.wgsl`](globe.wgsl) - the shader: stars, the globe with day and night textures, clouds, ocean glint and atmosphere, tone mapping, route glow, vignette, overlay compositing, fade and dither
- [`prep.py`](prep.py) - downloads the textures and builds the texture cache

Frames are encoded to H.264 (BT.709) with the hardware encoder by default.

## Credits

Textures from NASA Visible Earth: Blue Marble Next Generation (November 2004, topography and bathymetry), Black Marble 2016 night lights, cloud composite and GEBCO bathymetry (used as the water mask). Public domain, courtesy of NASA.
