"""Download the NASA source textures and build the texture cache.

    uv run python prep.py

Sources that are already present (and match their checksum) are not downloaded
again. The cache is a mip pyramid of raw .npy arrays per texture, which the
renderer uploads to the GPU.
"""

import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ASSETS = Path(__file__).parent / "assets"
CACHE = ASSETS / "cache"
NASA = "https://eoimages.gsfc.nasa.gov/images/imagerecords"

SOURCES = {
    # name: (file, url, sha256, mode, base width)
    "day": (
        # Blue Marble Next Generation, November 2004, with topography and bathymetry
        "world.topo.bathy.200411.3x21600x10800.jpg",
        f"{NASA}/73000/73884/world.topo.bathy.200411.3x21600x10800.jpg",
        "dbe2977b539562c4555b7df1fbc5b3313cdc4a0685fd8d5fe63dca2e1486ea21",
        "RGB",
        21600,
    ),
    "night": (
        # Black Marble 2016, city lights
        "BlackMarble_2016_3km.jpg",
        f"{NASA}/144000/144898/BlackMarble_2016_3km.jpg",
        "230aac448ae68c358be433dd518888cccb3a85ccf66f7b44326441c324ad6725",
        "RGB",
        10800,
    ),
    "clouds": (
        "cloud_combined_8192.tif",
        f"{NASA}/57000/57747/cloud_combined_8192.tif",
        "d137775d8966ab8d443fd5126dc6e7ad72072bc1ed50555c5818d221735daf0f",
        "L",
        8192,
    ),
    "water": (
        # GEBCO bathymetry, used as a land/water mask for the ocean glint
        "gebco_08_rev_bath_21600x10800.png",
        f"{NASA}/73000/73963/gebco_08_rev_bath_21600x10800.png",
        "b3a67076fccfbee73ebe437fd66a8ed4969fbc5ce949033b50675d43f379e403",
        "L",
        10800,
    ),
}
MIN_WIDTH = 1024
RETRIES = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, digest: str) -> None:
    """Download with resume and retries (the NASA server drops connections now and then)."""
    if dest.exists() and sha256(dest) == digest:
        print(f"{dest.name}: ok")
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # server ignored the range, start over
                    have = 0
                total = have + int(resp.headers.get("Content-Length", 0))
                with part.open("ab" if have else "wb") as f:
                    done = have
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r{dest.name}: {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
            print()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"\n{dest.name}: attempt {attempt} failed ({e}), retrying")
            time.sleep(2 * attempt)
    else:
        sys.exit(f"{dest.name}: download failed, try again later or fetch {url} manually into {ASSETS}/")
    if sha256(part) != digest:
        part.unlink()
        sys.exit(f"{dest.name}: checksum mismatch, the file was removed - run prep.py again")
    part.rename(dest)


def load(name: str) -> Image.Image:
    file, _, _, mode, width = SOURCES[name]
    im = Image.open(ASSETS / file)
    if name == "water":
        # GEBCO bathymetry: land is exactly 255, the ocean is darker.
        a = np.asarray(im.convert("L"))
        im = Image.fromarray(np.where(a < 255, 255, 0).astype(np.uint8))
    im = im.convert(mode)
    if im.width != width:
        im = im.resize((width, width // 2), Image.Resampling.LANCZOS)
    return im


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for file, url, digest, *_ in SOURCES.values():
        download(url, ASSETS / file, digest)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        im = load(name)
        level = 0
        while im.width >= MIN_WIDTH:
            out = CACHE / f"{name}_{level}.npy"
            np.save(out, np.ascontiguousarray(np.asarray(im)))
            print(f"{out.name}: {im.width}x{im.height}")
            im = im.reduce(2)
            level += 1


if __name__ == "__main__":
    main()
