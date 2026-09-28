"""Render a 3D globe fly-around animation described by a script (TOML).

The globe is ray traced on the GPU (globe.wgsl via gpu.py: NASA Blue Marble
day texture, Black Marble night lights, clouds, ocean glint, atmosphere), the
routes, markers and text are drawn with Pillow in worker processes, and frames
are encoded with the hardware H.264 encoder (VideoToolbox).

    uv run python prep.py                                               # once, builds assets/cache
    uv run python render.py scripts/around-the-world.toml --stills 3,20 # quick look at a few frames
    uv run python render.py scripts/around-the-world.toml --preset 4k   # final video

See scripts/*.toml for the script format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from gpu import GlobeGPU, sun_dir

ROOT = Path(__file__).parent
CACHE = ROOT / "assets" / "cache"
OUT = ROOT / "out"

EARTH_KM = 6371.0
FOV_SHORT = math.radians(30.0)  # field of view across the shorter frame side
TANH = math.tan(FOV_SHORT / 2)

# Defaults of the [timing] table of a script, in seconds
TIMING = {
    "intro": True,  # title card and zoom in from space
    "intro_duration": 5.5,
    "outro": True,  # rise over the North Pole and show the stats
    "outro_duration": 11.5,
    "start_hold": 1.4,  # on the first city before take-off
    "hold": 2.0,  # at each destination
    "transfer_hold": 0.7,  # at connections, unless flying through them
    "final_hold": 1.2,  # at the last city (before the outro, if any)
    "fly_through_transfers": False,  # continue from a connection without stopping
    "flight_speed": 1.0,  # multiplier of the flight speed
    "fade_in": 0.6,  # from black
    "fade_out": 1.0,  # to black
}

# Camera distances from the Earth's centre, in Earth radii
D_INTRO = 7.5
D_HOLD = 2.0
D_OUTRO = 4.7
OUTRO_LAT = 82.0
OUTRO_ORBIT = 140.0  # degrees of longitude the camera orbits during the outro

GOLD = (255, 198, 109)
WHITE = (255, 255, 255)

FONT_AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
FONT_DIN = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
AVENIR = {"bold": 0, "demi": 2, "medium": 5, "regular": 7, "heavy": 8}


# ---------------------------------------------------------------------- script


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_script(path: Path) -> dict:
    """Read a TOML script; files listed in `include` are merged underneath it."""
    data = tomllib.loads(path.read_text())
    merged: dict = {}
    for inc in data.pop("include", []):
        merged = _merge(merged, load_script(path.parent / inc))
    return _merge(merged, data)


def resolve_script(path: Path) -> dict:
    script = load_script(path)
    unknown = set(script.get("timing", {})) - set(TIMING)
    if unknown:
        sys.exit(f"{path}: unknown [timing] keys: {', '.join(sorted(unknown))}")
    script["timing"] = TIMING | script.get("timing", {})
    for key in ("stops", "legs"):
        if not script.get(key):
            sys.exit(f"{path}: the script has no {key}")
    for leg in script["legs"]:
        for end in ("from", "to"):
            if leg[end] not in script["stops"]:
                sys.exit(f"{path}: unknown stop {leg[end]!r} in legs")
    return script


# ---------------------------------------------------------------- math helpers


def ll2v(lat: float, lon: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.sin(la), -math.cos(la) * math.sin(lo)])


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def slerp(a: np.ndarray, b: np.ndarray, t):
    """Spherical interpolation; t may be a scalar or a 1D array."""
    w = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    t = np.asarray(t, dtype=np.float64)
    if w < 1e-9:
        return np.broadcast_to(a, t.shape + (3,)).copy()
    s = math.sin(w)
    return (np.sin((1 - t) * w)[..., None] * a + np.sin(t * w)[..., None] * b) / s


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = clamp01((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + (b - a) * t


# -------------------------------------------------------------------- timeline


@dataclass
class Stop:
    code: str
    name: str
    country: str
    v: np.ndarray
    label: str


@dataclass
class Leg:
    a: Stop
    b: Stop
    date: str | None
    transfer: bool
    angle: float
    km: float
    duration: float
    height: float
    bump: float
    t0: float = 0.0
    t1: float = 0.0
    m0: float = 0.0  # normalised speed at take-off (non-zero when flying through a connection)
    m1: float = 0.0  # normalised speed at landing
    first: bool = True  # first leg of its leap
    last: bool = True  # last leg of its leap
    leap_t0: float = 0.0
    leap_t1: float = 0.0

    def point(self, s):
        """3D position on the (raised) flight arc at progress s in [0, 1]."""
        s = np.asarray(s, dtype=np.float64)
        return slerp(self.a.v, self.b.v, s) * (1 + self.height * np.sin(np.pi * s))[..., None]

    def progress(self, t: float) -> float:
        """Progress along the arc at time t (eased)."""
        u = min(1.0, max(0.0, (t - self.t0) / self.duration))
        if self.m0 == 0 and self.m1 == 0:
            return float(smoother(u))
        # cubic Hermite from speed m0 to m1 (monotonic for m <= 3)
        return (u**3 - 2 * u**2 + u) * self.m0 + (3 * u**2 - 2 * u**3) + (u**3 - u**2) * self.m1


@dataclass
class Visit:
    stop: Stop
    t_arrive: float
    t_depart: float
    transfer: bool


@dataclass
class Camera:
    F: np.ndarray  # unit vector from the Earth's centre towards the camera
    D: float  # distance from the Earth's centre
    sx: float = 0.0  # lens shift, in half-short-side units
    sy: float = 0.0


class Timeline:
    def __init__(self, script: dict, vertical: bool):
        tm = script["timing"]
        self.title = script.get("title", "")
        self.subtitle = script.get("subtitle", "")
        self.vertical = vertical
        self.intro = bool(tm["intro"])
        self.outro = bool(tm["outro"])
        self.intro_dur = float(tm["intro_duration"]) if self.intro else 0.0
        self.outro_dur = float(tm["outro_duration"]) if self.outro else 0.0
        self.fade_in = float(tm["fade_in"])
        self.fade_out = float(tm["fade_out"])
        stops = {
            code: Stop(code, s["name"], s["country"], ll2v(s["lat"], s["lon"]), s.get("label", "right"))
            for code, s in script["stops"].items()
        }
        self.stop_ll = {code: (s["lat"], s["lon"]) for code, s in script["stops"].items()}
        self.legs: list[Leg] = []
        for leg in script["legs"]:
            a, b = stops[leg["from"]], stops[leg["to"]]
            angle = math.acos(max(-1.0, min(1.0, float(np.dot(a.v, b.v)))))
            km = angle * EARTH_KM
            self.legs.append(
                Leg(
                    a=a,
                    b=b,
                    date=leg.get("date"),
                    transfer=bool(leg.get("transfer")),
                    angle=angle,
                    km=km,
                    duration=(2.0 + 0.055 * math.sqrt(km)) / float(tm["flight_speed"]),
                    height=min(0.075, 0.006 + 0.05 * angle),
                    bump=min(2.4, 1.75 * angle),
                )
            )

        # A leap is a run of legs flown without stopping: with fly_through_transfers
        # it continues through connections, otherwise every leg is its own leap.
        self.leaps: list[list[Leg]] = []
        for i, leg in enumerate(self.legs):
            if self.leaps and tm["fly_through_transfers"] and self.legs[i - 1].transfer:
                self.leaps[-1].append(leg)
            else:
                self.leaps.append([leg])
        for leap in self.leaps:
            for x, y in zip(leap, leap[1:]):
                # pass through the connection at a reduced but non-zero speed
                v = 0.6 * min(x.km / x.duration, y.km / y.duration)
                x.m1 = v * x.duration / x.km
                y.m0 = v * y.duration / y.km
                x.last = y.first = False

        t = self.intro_dur + float(tm["start_hold"])
        self.visits = [Visit(self.legs[0].a, self.intro_dur - 1.4, 0.0, False)]
        for k, leap in enumerate(self.leaps):
            leap_t0 = t
            for leg in leap:
                leg.t0, leg.t1 = t, t + leg.duration
                leg.leap_t0 = leap_t0
                self.visits[-1].t_depart = leg.t0
                self.visits.append(Visit(leg.b, leg.t1, math.inf, leg.transfer and leg is not self.legs[-1]))
                t = leg.t1
            for leg in leap:
                leg.leap_t1 = t
            if k == len(self.leaps) - 1:
                t += float(tm["final_hold"])
            else:
                t += float(tm["transfer_hold"] if leap[-1].transfer else tm["hold"])
        self.outro_t0 = t
        self.total = t + self.outro_dur
        self.total_km = sum(leg.km for leg in self.legs)
        self.countries = len({v.stop.country for v in self.visits})
        self.intro_from = ll2v(18.0, self.stop_ll[self.legs[0].a.code][1] - 75.0)

    # ---- camera

    def camera(self, t: float) -> Camera:
        if t < self.intro_dur:
            e = float(smoother(t / self.intro_dur))
            F = slerp(self.intro_from, self.legs[0].a.v, e)
            # ease the zoom a bit later than the rotation
            D = lerp(D_INTRO, D_HOLD, float(smoother((t / self.intro_dur) ** 0.8)))
            return Camera(F, D)
        for leap in self.leaps:
            if t < leap[0].t0:
                return Camera(leap[0].a.v, D_HOLD)
            if t >= leap[-1].t1:
                continue
            if len(leap) == 1:
                leg = leap[0]
                u = (t - leg.t0) / leg.duration
                sc = float(smoother((u - 0.05) / 0.95))
                F = slerp(leg.a.v, leg.b.v, sc)
                return Camera(F, D_HOLD + leg.bump * math.sin(math.pi * u) ** 2)
            # follow the plane with a temporal blur, so the camera rounds the connections
            offsets = np.linspace(-0.6, 0.6, 13)
            weights = np.exp(-0.5 * (offsets / 0.25) ** 2)
            pts = [self.ground(leap, min(leap[-1].t1, max(leap[0].t0, t + o))) for o in offsets]
            F = normalize((weights[:, None] * np.array(pts)).sum(axis=0))
            leg = next(g for g in leap if t < g.t1)
            u = (t - leg.t0) / leg.duration
            return Camera(F, D_HOLD + leg.bump * math.sin(math.pi * u) ** 2)
        if t < self.outro_t0 or not self.outro:
            return Camera(self.legs[-1].b.v, D_HOLD)
        u = (t - self.outro_t0) / self.outro_dur
        lat0, lon0 = self.stop_ll[self.legs[-1].b.code]
        lat = lerp(lat0, OUTRO_LAT, float(smoother(u / 0.5)))
        lon = lon0 + OUTRO_ORBIT * float(smoother(u))
        D = lerp(D_HOLD, D_OUTRO, float(smoother(u / 0.55)))
        k = float(smoother((u - 0.1) / 0.5))
        if self.vertical:
            return Camera(ll2v(lat, lon), D, 0.0, -0.55 * k)
        return Camera(ll2v(lat, lon), D, 0.78 * k, 0.0)

    @staticmethod
    def ground(leap: list[Leg], t: float) -> np.ndarray:
        leg = next((g for g in leap if t <= g.t1), leap[-1])
        return slerp(leg.a.v, leg.b.v, leg.progress(t))

    # ---- flight state

    def plane(self, t: float):
        """(leg index, progress) of the plane in the air, or None."""
        for i, leg in enumerate(self.legs):
            if leg.t0 <= t < leg.t1:
                return i, leg.progress(t)
        return None

    def plane_point(self, leg: Leg, t: float) -> np.ndarray:
        """3D plane position at time t, clamped to the leap of the given leg."""
        t = min(leg.leap_t1, max(leg.leap_t0, t))
        g = next(g for g in self.legs if g.leap_t0 == leg.leap_t0 and t <= g.t1)
        return g.point(np.array([g.progress(t)]))[0]

    def legs_done(self, t: float) -> int:
        return sum(1 for leg in self.legs if t >= leg.t1)

    def km_flown(self, t: float) -> float:
        km = 0.0
        for leg in self.legs:
            if t >= leg.t1:
                km += leg.km
            elif t > leg.t0:
                km += leg.km * leg.progress(t)
        return km

    def hud_leg(self, t: float) -> tuple[int, float]:
        """Leg shown in the HUD and the alpha of its crossfade."""
        idx, switch = 0, -math.inf
        for i, leg in enumerate(self.legs):
            at = leg.t0 - (0.6 if leg.first else 0.2)
            if t >= at:
                idx, switch = i, at
        alpha = 1.0 if idx == 0 else min(1.0, abs(t - switch) / 0.25)
        return idx, alpha


# -------------------------------------------------------------------- renderer


PLANE_HALF = [
    (0.50, 0.0),
    (0.46, 0.035),
    (0.36, 0.048),
    (0.10, 0.05),
    (-0.08, 0.46),
    (-0.17, 0.46),
    (-0.08, 0.05),
    (-0.33, 0.045),
    (-0.43, 0.19),
    (-0.50, 0.19),
    (-0.47, 0.02),
    (-0.50, 0.0),
]
PLANE = PLANE_HALF + [(x, -y) for x, y in reversed(PLANE_HALF[1:-1])]


class Overlays:
    """The 2D layers drawn on the CPU: routes, markers, plane, labels, HUD, cards."""

    def __init__(self, script: dict, width: int, height: int):
        self.W, self.H = width, height
        self.S = min(width, height)
        self.half = self.S / 2
        self.vertical = height > width
        self.tl = Timeline(script, self.vertical)
        self.fonts: dict = {}

    # ---- fonts

    def font(self, kind: str, size: float) -> ImageFont.FreeTypeFont:
        key = (kind, int(round(size)))
        if key not in self.fonts:
            if kind == "din":
                self.fonts[key] = ImageFont.truetype(FONT_DIN, key[1])
            else:
                self.fonts[key] = ImageFont.truetype(FONT_AVENIR, key[1], index=AVENIR[kind])
        return self.fonts[key]

    # ---- camera basis and projection

    def basis(self, cam: Camera):
        C = cam.F * cam.D
        f = -cam.F
        r = normalize(np.cross(f, np.array([0.0, 1.0, 0.0])))
        u = np.cross(r, f)
        return C, f, r, u

    def project(self, pts: np.ndarray, cam: Camera, basis):
        """Project Nx3 world points; returns x, y (pixels) and a visibility mask."""
        C, f, r, u = basis
        v = pts - C
        z = v @ f
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        x = (v @ r) / zs / TANH + cam.sx
        y = (v @ u) / zs / TANH + cam.sy
        px = x * self.half + self.W / 2
        py = -y * self.half + self.H / 2
        dist = np.linalg.norm(v, axis=1)
        dn = v / dist[:, None]
        b = dn @ C
        h2 = cam.D**2 - b**2
        t_enter = -b - np.sqrt(np.maximum(1 - h2, 0))
        hidden = (h2 < 1) & (t_enter > 0) & (t_enter < dist - 1e-3)
        return px, py, (~hidden) & (z > 0)

    def facing(self, p: np.ndarray, cam: Camera) -> float:
        """Soft visibility for a point on the surface (fades out at the limb)."""
        C = cam.F * cam.D
        c = float(np.dot(p, normalize(C - p)))
        return float(smoothstep(0.02, 0.15, c))

    # ---- vector overlay (routes, markers, plane)

    def arc_screen(self, leg: Leg, s_to: float, cam: Camera, basis, n: int = 360):
        s = np.linspace(0, s_to, max(2, int(n * max(s_to, 0.02))))
        px, py, vis = self.project(leg.point(s), cam, basis)
        return px, py, vis

    @staticmethod
    def runs(px, py, vis, scale):
        """Split a projected polyline into its visible runs."""
        out, cur = [], []
        for x, y, ok in zip(px, py, vis):
            if ok:
                cur.append((x * scale, y * scale))
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return [r for r in out if len(r) >= 2]

    def draw_overlay(self, t: float, cam: Camera, basis) -> tuple[Image.Image, Image.Image]:
        """The crisp RGBA layer and the quarter resolution route glow."""
        tl = self.tl
        S = self.S
        ss = 2
        layer = Image.new("RGBA", (self.W * ss, self.H * ss), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        gs = 0.25
        glow = Image.new("RGB", (int(self.W * gs), int(self.H * gs)), (0, 0, 0))
        dg = ImageDraw.Draw(glow)

        lw = max(2, int(round(0.0032 * S * ss)))
        glw = max(2, int(round(0.012 * S * gs)))
        plane = tl.plane(t)
        done = tl.legs_done(t)
        outro_k = float(smoothstep(tl.outro_t0, tl.outro_t0 + 2.0, t))

        def polyline(pts_runs, width, fill, draw):
            for run in pts_runs:
                draw.line(run, fill=fill, width=width, joint="curve")

        # upcoming leg preview (dashed) shortly before and during the flight
        for i, leg in enumerate(tl.legs):
            a = float(smoothstep(leg.leap_t0 - 1.0, leg.leap_t0 - 0.2, t)) * (1 - float(smoothstep(leg.t1 - 0.3, leg.t1, t)))
            if a <= 0:
                continue
            px, py, vis = self.arc_screen(leg, 1.0, cam, basis, n=240)
            dash = (np.arange(len(px)) // 4) % 2 == 0
            polyline(self.runs(px, py, vis & dash, ss), max(1, int(lw * 0.6)), GOLD + (int(160 * a),), dr)

        # completed legs and the leg in the air
        for i, leg in enumerate(tl.legs):
            if i < done:
                s_to = 1.0
            elif plane and plane[0] == i:
                s_to = plane[1]
            else:
                continue
            if s_to <= 0.001:
                continue
            px, py, vis = self.arc_screen(leg, s_to, cam, basis)
            glow_a = 0.55 + 0.45 * outro_k if i < done else 1.0
            polyline(self.runs(px, py, vis, gs), glw, tuple(int(c * glow_a * 0.8) for c in GOLD), dg)
            polyline(self.runs(px, py, vis, ss), lw, GOLD + (235,), dr)
            if not (i < done):
                # bright comet head behind the plane
                k = max(2, int(len(px) * 0.18))
                polyline(self.runs(px[-k:], py[-k:], vis[-k:], ss), max(1, int(lw * 0.55)), (255, 244, 220, 230), dr)

        # markers
        for vi, visit in enumerate(tl.visits):
            p = visit.stop.v
            appear = float(smoothstep(visit.t_arrive - 0.05, visit.t_arrive + 0.25, t))
            if vi == 0:
                appear = float(smoothstep(tl.intro_dur - 2.0, tl.intro_dur - 1.0, t))
            face = self.facing(p, cam)
            if appear <= 0 or face <= 0:
                continue
            px, py, vis = self.project(p[None], cam, basis)
            x, y = px[0] * ss, py[0] * ss
            small = visit.transfer
            rad = (0.0052 if small else 0.0068) * S * ss
            a = appear * face
            # arrival pulse
            dt = t - visit.t_arrive
            if 0 <= dt < 1.4 and vi > 0:
                k = dt / 1.4
                pr = rad * (1.5 + 5.0 * k)
                pa = int(200 * (1 - k) ** 2 * face)
                dr.ellipse((x - pr, y - pr, x + pr, y + pr), outline=GOLD + (pa,), width=max(1, lw // 2))
            ring = rad * 1.75
            dr.ellipse((x - ring, y - ring, x + ring, y + ring), fill=GOLD + (int(90 * a),))
            dr.ellipse((x - rad, y - rad, x + rad, y + rad), fill=WHITE + (int(255 * a),))
            gx, gy = px[0] * gs, py[0] * gs
            gr = rad / ss * gs * 3
            dg.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=tuple(int(c * 0.6 * a) for c in GOLD))

        # destination target while flying
        if plane:
            leg = tl.legs[plane[0]]
            u = (t - leg.t0) / leg.duration
            face = self.facing(leg.b.v, cam)
            if face > 0:
                px, py, vis = self.project(leg.b.v[None], cam, basis)
                x, y = px[0] * ss, py[0] * ss
                k = (u * leg.duration / 1.1) % 1.0
                pr = 0.008 * S * ss * (1 + 1.6 * k)
                a = int(220 * (1 - k) * face * float(smoothstep(0, 0.08, u)))
                dr.ellipse((x - pr, y - pr, x + pr, y + pr), outline=GOLD + (a,), width=max(1, lw // 2))
                cr = 0.0035 * S * ss
                dr.ellipse((x - cr, y - cr, x + cr, y + cr), fill=GOLD + (int(230 * face),))

        # the plane
        if plane:
            i, s = plane
            leg = tl.legs[i]
            u = (t - leg.t0) / leg.duration
            a = (float(smoothstep(0.0, 0.06, u)) if leg.first else 1.0) * (
                1 - float(smoothstep(0.94, 1.0, u)) if leg.last else 1.0
            )
            ds = 0.004
            pts = [
                leg.point(np.array([s]))[0],
                tl.plane_point(leg, t - 0.08),
                tl.plane_point(leg, t + 0.08),
                *leg.point(np.array([max(0.0, s - ds), min(1.0, s + ds)])),
            ]
            px, py, vis = self.project(np.array(pts), cam, basis)
            if vis[0] and a > 0:
                x, y = px[0], py[0]
                k = 1 if math.hypot(px[2] - px[1], py[2] - py[1]) > 0.5 else 3
                ang = math.atan2(py[k + 1] - py[k], px[k + 1] - px[k])
                size = 0.046 * S * (0.85 + 0.15 * math.sin(math.pi * s))
                ca, sa = math.cos(ang), math.sin(ang)
                poly = [((x + (qx * ca - qy * sa) * size) * ss, (y + (qx * sa + qy * ca) * size) * ss) for qx, qy in PLANE]
                shadow = [(qx + 0.004 * S * ss, qy + 0.006 * S * ss) for qx, qy in poly]
                dr.polygon(shadow, fill=(0, 0, 0, int(90 * a)))
                dr.polygon(poly, fill=WHITE + (int(255 * a),), outline=(40, 30, 10, int(160 * a)), width=max(1, ss))
                gr = size * gs * 0.3
                dg.ellipse((x * gs - gr, y * gs - gr, x * gs + gr, y * gs + gr), fill=tuple(int(c * 0.5 * a) for c in GOLD))

        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1.0, 0.006 * S * gs)))
        return layer.reduce(ss), glow

    # ---- text

    def text(self, dr: ImageDraw.ImageDraw, xy, s: str, font, fill, tracking: float = 0.0, anchor: str = "ls"):
        """Draw text with letter spacing (tracking in em); anchor h in l/m/r."""
        x, y = xy
        if tracking == 0:
            dr.text((x, y), s, font=font, fill=fill, anchor=anchor)
            return font.getlength(s)
        sp = tracking * font.size
        widths = [font.getlength(ch) for ch in s]
        total = sum(widths) + sp * (len(s) - 1)
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        for ch, w in zip(s, widths):
            dr.text((x, y), ch, font=font, fill=fill, anchor="l" + anchor[1])
            x += w + sp
        return total

    def plane_icon(self, dr, x, y, size, fill, ang=0.0):
        ca, sa = math.cos(ang), math.sin(ang)
        dr.polygon([(x + (qx * ca - qy * sa) * size, y + (qx * sa + qy * ca) * size) for qx, qy in PLANE], fill=fill)

    def in_hud(self, x0: float, x1: float, y: float) -> float:
        """1 when a label spanning x0..x1 at height y overlaps a HUD block, fading to 0 nearby."""
        S, H, m = self.S, self.H, 0.05 * self.S
        if self.vertical:
            boxes = [(0, 0, m + 0.42 * S, m + 0.1 * H + 0.33 * S)]
        else:
            boxes = [(0, 0, m + 0.42 * S, m + 0.16 * S), (0, H - m - 0.14 * S, m + 0.42 * S, H)]
        lo, hi = min(x0, x1), max(x0, x1)
        d = math.inf
        for bx0, by0, bx1, by1 in boxes:
            dx = max(bx0 - hi, lo - bx1, 0.0)
            dy = max(by0 - y, y - by1, 0.0)
            d = min(d, math.hypot(dx, dy))
        return 1 - float(smoothstep(0.0, 0.05 * S, d))

    def draw_text(self, t: float, cam: Camera, basis) -> Image.Image:
        tl = self.tl
        S, W, H = self.S, self.W, self.H
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)

        def rgba(c, a):
            return c + (int(max(0, min(1, a)) * 255),)

        hud_in = float(smoothstep(tl.intro_dur - 0.8, tl.intro_dur + 0.2, t)) if tl.intro else 1.0
        hud_a = hud_in * (1 - float(smoothstep(tl.outro_t0 + 0.5, tl.outro_t0 + 2.0, t)))

        # ---- city labels
        outro_hide = 1 - float(smoothstep(tl.outro_t0 + 1.5, tl.outro_t0 + 3.0, t))
        seen: dict[str, float] = {}
        big: dict[str, float] = {}
        for vi, visit in enumerate(tl.visits):
            code = visit.stop.code
            arrive = tl.intro_dur - 1.4 if vi == 0 else visit.t_arrive
            seen[code] = max(seen.get(code, 0.0), float(smoothstep(arrive, arrive + 0.5, t)))
            if not visit.transfer:
                a = float(smoothstep(arrive + 0.1, arrive + 0.6, t))
                if visit.t_depart < math.inf:
                    a *= 1 - float(smoothstep(visit.t_depart + 0.4, visit.t_depart + 0.9, t))
                big[code] = max(big.get(code, 0.0), a * outro_hide)
        stops = {v.stop.code: v.stop for v in tl.visits}
        transfer_only = {c for c in stops if all(v.transfer for v in tl.visits if v.stop.code == c)}
        for code, stop in stops.items():
            face = self.facing(stop.v, cam)
            if face <= 0:
                continue
            px, py, _ = self.project(stop.v[None], cam, basis)
            x, y = float(px[0]), float(py[0])
            side = 1 if stop.label == "right" else -1
            anchor_h = "l" if side > 0 else "r"
            ab = big.get(code, 0.0) * face
            asmall = seen.get(code, 0.0) * (1 - big.get(code, 0.0)) * face
            if code in transfer_only:
                asmall *= float(smoothstep(3.0, 2.4, cam.D)) * outro_hide
            # keep labels out of the HUD corners
            clear = 1 - hud_a * self.in_hud(x, x + side * 0.2 * S, y)
            ab *= clear
            asmall *= clear
            if ab > 0.003:
                slide = (1 - ab) * 0.012 * S * side
                nx = x + side * 0.034 * S + slide
                name_f = self.font("demi", 0.046 * S)
                sub_f = self.font("demi", 0.0165 * S)
                self.text(dr, (nx, y + 0.012 * S), stop.name, name_f, rgba(WHITE, ab), anchor=anchor_h + "s")
                self.text(dr, (nx + side * 0.002 * S, y + 0.024 * S), stop.country.upper(), sub_f, rgba(GOLD, ab), 0.22, anchor_h + "t")
            if asmall > 0.003:
                nx = x + side * 0.016 * S
                f = self.font("demi", 0.022 * S)
                self.text(dr, (nx, y), stop.name, f, rgba(WHITE, asmall * 0.85), anchor=anchor_h + "m")

        # ---- HUD
        m = 0.05 * S
        if hud_a > 0:
            idx, cross = tl.hud_leg(t)
            leg = tl.legs[idx]
            a = hud_a * cross
            top = m + (0.1 * H if self.vertical else 0)
            small = self.font("demi", 0.017 * S)
            codes = self.font("din", 0.058 * S)
            self.text(dr, (m, top + 0.017 * S), f"FLIGHT {idx + 1} OF {len(tl.legs)}", small, rgba(GOLD, hud_a), 0.25)
            y = top + 0.085 * S
            w = self.text(dr, (m, y), leg.a.code, codes, rgba(WHITE, a))
            self.plane_icon(dr, m + w + 0.042 * S, y - 0.021 * S, 0.042 * S, rgba(WHITE, a))
            self.text(dr, (m + w + 0.084 * S, y), leg.b.code, codes, rgba(WHITE, a))
            info = leg.date.upper() if leg.date else ""
            if leg.transfer:
                info = f"{info}  ·  CONNECTION" if info else "CONNECTION"
            if info:
                self.text(dr, (m, y + 0.042 * S), info, small, rgba(WHITE, a * 0.8), 0.25)

            label_y = H - m - 0.075 * S if not self.vertical else top + 0.23 * S
            self.text(dr, (m, label_y), "DISTANCE FLOWN", small, rgba(GOLD, hud_a), 0.25)
            km = f"{int(round(tl.km_flown(t))):,}".replace(",", " ")
            num = self.font("din", 0.058 * S)
            w = self.text(dr, (m, label_y + 0.068 * S), km, num, rgba(WHITE, hud_a))
            self.text(dr, (m + w + 0.012 * S, label_y + 0.068 * S), "km", self.font("demi", 0.03 * S), rgba(WHITE, hud_a * 0.8))

        # ---- title card
        ta = float(smoothstep(0.5, 1.4, t)) * (1 - float(smoothstep(3.6, 4.5, t))) if tl.intro else 0.0
        if ta > 0:
            cx, cy = W / 2, H / 2 - (0.05 * S if not self.vertical else 0.3 * S)
            prog = float(smoother((t - 0.5) / 4.0))
            size = (0.085 if not self.vertical else 0.1) * S
            title_f = self.font("heavy", size)
            # fit the title (at its widest letter spacing) into 90 % of the frame width
            wide = sum(title_f.getlength(ch) for ch in tl.title.upper()) + 0.16 * size * (len(tl.title) - 1)
            if wide > 0.9 * W:
                title_f = self.font("heavy", size * 0.9 * W / wide)
            self.text(dr, (cx, cy), tl.title.upper(), title_f, rgba(WHITE, ta), 0.16 - 0.04 * prog, "ms")
            lw = 0.06 * S * float(smoother((t - 0.9) / 1.0))
            dr.rectangle((cx - lw, cy + 0.03 * S, cx + lw, cy + 0.034 * S), fill=rgba(GOLD, ta))
            self.text(dr, (cx, cy + 0.095 * S), tl.subtitle.upper(), self.font("demi", 0.026 * S), rgba(WHITE, ta * 0.9), 0.3, "ms")

        # ---- end card
        ea = float(smoothstep(tl.outro_t0 + 4.4, tl.outro_t0 + 5.4, t)) if tl.outro else 0.0
        if ea > 0:
            if self.vertical:
                x, y0, anchor = W / 2, 0.1 * H, "m"
            else:
                x, y0, anchor = 0.075 * W, 0.2 * H, "l"
            title_f = self.font("heavy", 0.052 * S)
            self.text(dr, (x, y0 + 0.06 * S), tl.title.upper(), title_f, rgba(WHITE, ea), 0.1, anchor + "s")
            self.text(dr, (x, y0 + 0.115 * S), tl.subtitle.upper(), self.font("demi", 0.022 * S), rgba(GOLD, ea), 0.3, anchor + "s")
            stats = [
                (str(len(tl.legs)), "FLIGHTS"),
                (str(tl.countries), "COUNTRIES"),
                (f"{int(round(tl.total_km, -2)):,}".replace(",", " "), "KILOMETERS"),
            ]
            big_f = self.font("din", 0.085 * S)
            lab_f = self.font("demi", 0.019 * S)
            for k, (value, label) in enumerate(stats):
                start = tl.outro_t0 + 5.2 + 0.35 * k
                a = float(smoothstep(start, start + 0.6, t))
                if a <= 0:
                    continue
                if self.vertical:
                    sx = W / 2 + (k - 1) * 0.3 * W
                    sy = y0 + 0.28 * S
                    anc = "m"
                else:
                    sx = x
                    sy = y0 + 0.28 * S + k * 0.155 * S
                    anc = "l"
                dy = (1 - a) * 0.02 * S
                self.text(dr, (sx, sy + dy), value, big_f, rgba(WHITE, a), 0, anc + "s")
                self.text(dr, (sx + (0.003 * S if anc == "l" else 0), sy + 0.042 * S + dy), label, lab_f, rgba(GOLD, a), 0.3, anc + "s")
        return layer

    # ---- one frame

    def layers(self, t: float) -> tuple[bytes, bytes]:
        """RGBA overlay (full size) and RGBA glow (quarter size) for time t."""
        cam = self.tl.camera(t)
        basis = self.basis(cam)
        vec, glow = self.draw_overlay(t, cam, basis)
        txt = self.draw_text(t, cam, basis)
        shadow = Image.new("RGBA", txt.size, (0, 0, 0, 0))
        shadow.putalpha(txt.getchannel("A").point(lambda v: v * 0.55))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=0.004 * self.S))
        vec.alpha_composite(shadow)
        vec.alpha_composite(txt)
        return vec.tobytes(), glow.convert("RGBA").tobytes()


def gpu_frame(gpu: GlobeGPU, ov: Overlays, t: float, layers: tuple[bytes, bytes], seed: int) -> np.ndarray:
    """Composite the final (H, W, 4) frame on the GPU."""
    tl = ov.tl
    cam = tl.camera(t)
    basis = ov.basis(cam)
    fade = min(
        1.0,
        t / tl.fade_in if tl.fade_in > 0 else 1.0,
        (tl.total - t) / tl.fade_out if tl.fade_out > 0 else 1.0,
    )
    return gpu.render(basis, cam, sun_dir(basis), *layers, max(0.0, fade), seed, TANH)


# ------------------------------------------------------------ multiprocessing

_OV: Overlays | None = None


def _init(script, width, height):
    global _OV
    _OV = Overlays(script, width, height)


def _layers(t: float) -> tuple[bytes, bytes]:
    return _OV.layers(t)


SEGMENT = 150  # frames per resumable video segment
PRESETS = {"preview": (960, 540), "1080": (1920, 1080), "4k": (3840, 2160)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", type=Path, help="script file, e.g. scripts/around-the-world.toml")
    ap.add_argument("--preset", choices=PRESETS, default="preview")
    ap.add_argument("--vertical", action="store_true", help="9:16 frame for Reels/Shorts/TikTok")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3, help="processes drawing the 2D overlays")
    ap.add_argument("--stills", help="comma separated times (s) to render as PNG instead of a video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--encoder", choices=("vt", "x264"), default="vt", help="vt: hardware (VideoToolbox), x264: software")
    ap.add_argument("--quality", type=int, default=75, help="VideoToolbox quality, 1-100")
    ap.add_argument("--crf", type=int, default=15, help="x264 quality (lower is better)")
    ap.add_argument("--out", help="output file")
    args = ap.parse_args()

    script = resolve_script(args.script)
    name = args.script.stem
    w, h = PRESETS[args.preset]
    if args.vertical:
        w, h = h, w
    OUT.mkdir(exist_ok=True)
    tl = Timeline(script, args.vertical)
    print(f"{w}x{h}, duration {tl.total:.1f}s, {tl.total_km:,.0f} km, {len(tl.legs)} flights")

    gpu = GlobeGPU(w, h)
    if args.stills:
        ov = Overlays(script, w, h)
        for s in args.stills.split(","):
            t = float(s)
            t0 = time.time()
            img = gpu_frame(gpu, ov, t, ov.layers(t), int(t * args.fps))
            p = OUT / "stills" / f"{name}_{args.preset}{'_v' if args.vertical else ''}_{t:06.2f}.png"
            p.parent.mkdir(exist_ok=True)
            Image.fromarray(img[..., :3]).save(p)
            print(f"{p} ({time.time() - t0:.1f}s)")
        return

    end = min(tl.total, args.end) if args.end else tl.total
    frames = list(range(int(args.start * args.fps), int(end * args.fps)))
    out = Path(args.out) if args.out else OUT / f"{name}-{args.preset}{'-vertical' if args.vertical else ''}.mp4"
    # Render in segments so an interrupted render can resume where it stopped.
    code = "".join((ROOT / f).read_text() for f in ("render.py", "gpu.py", "globe.wgsl"))
    enc_opts = f"{args.encoder} q{args.quality}" if args.encoder == "vt" else f"x264 crf{args.crf}"
    key = json.dumps(script, sort_keys=True) + code + f"{w}x{h}@{args.fps} {enc_opts}"
    parts_dir = OUT / "parts" / f"{out.stem}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    segments = [frames[i : i + SEGMENT] for i in range(0, len(frames), SEGMENT)]
    todo = [(k, seg) for k, seg in enumerate(segments) if not (parts_dir / f"{seg[0]:06d}.mp4").exists()]
    n_todo = sum(len(seg) for _, seg in todo)
    print(f"{len(segments) - len(todo)}/{len(segments)} segments already done, {n_todo} frames to render")
    t0 = time.time()
    done = 0
    if todo:
        ov = Overlays(script, w, h)
        with Pool(args.workers, initializer=_init, initargs=(script, w, h), maxtasksperchild=200) as pool:
            for k, seg in todo:
                part = parts_dir / f"{seg[0]:06d}.mp4"
                tmp = part.with_suffix(".tmp.mp4")
                enc = subprocess.Popen(encoder_cmd(w, h, args, tmp), stdin=subprocess.PIPE)
                for i, layers in zip(seg, pool.imap(_layers, [i / args.fps for i in seg], chunksize=1)):
                    enc.stdin.write(gpu_frame(gpu, ov, i / args.fps, layers, i).tobytes())
                    done += 1
                    if done % 10 == 0 or done == n_todo:
                        el = time.time() - t0
                        eta = el / done * (n_todo - done)
                        print(f"\rframe {done}/{n_todo}  {el:.0f}s elapsed, eta {eta:.0f}s   ", end="", flush=True)
                enc.stdin.close()
                if enc.wait() != 0:
                    sys.exit(f"ffmpeg failed on segment {part.name}")
                tmp.rename(part)
    listing = parts_dir / "parts.txt"
    listing.write_text("".join(f"file '{parts_dir / f'{seg[0]:06d}.mp4'}'\n" for seg in segments))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
         "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip
    print(f"\n{out}")


def encoder_cmd(w: int, h: int, args, out: Path) -> list[str]:
    if args.encoder == "vt":
        codec = ["-c:v", "h264_videotoolbox", "-q:v", str(args.quality), "-profile:v", "high", "-prio_speed", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-tune", "film",
                 "-profile:v", "high", "-x264-params", "aq-mode=3"]  # fmt: skip
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        *codec,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        str(out),
    ]  # fmt: skip


if __name__ == "__main__":
    main()
