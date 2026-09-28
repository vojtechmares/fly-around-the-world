"""GPU (Metal via wgpu) renderer of the globe and the final frame composite."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import wgpu

ROOT = Path(__file__).parent
CACHE = ROOT / "assets" / "cache"
SHADER = ROOT / "globe.wgsl"

TEX_USAGE = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST


def _levels(name: str) -> list[np.ndarray]:
    levels = []
    while (p := CACHE / f"{name}_{len(levels)}.npy").exists():
        levels.append(np.load(p, mmap_mode="r"))
    if not levels:
        sys.exit(f"missing {CACHE}/{name}_0.npy - run `uv run python prep.py` first")
    return levels


def _rgba(a: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = a
    out[..., 3] = 255
    return out


class GlobeGPU:
    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync(
            required_limits={"max-texture-dimension-2d": adapter.limits["max-texture-dimension-2d"]}
        )
        dev = self.device
        self.textures = self._upload_textures()
        self.glow_size = (int(width * 0.25), int(height * 0.25))
        self.glow = dev.create_texture(size=(*self.glow_size, 1), format="rgba8unorm", usage=TEX_USAGE)
        self.overlay = dev.create_texture(size=(width, height, 1), format="rgba8unorm", usage=TEX_USAGE)
        self.target = dev.create_texture(
            size=(width, height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.uniforms = dev.create_buffer(size=7 * 16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        def aniso(mode_u):
            return dev.create_sampler(
                address_mode_u=mode_u,
                address_mode_v="clamp-to-edge",
                mag_filter="linear",
                min_filter="linear",
                mipmap_filter="linear",
                max_anisotropy=16,
            )

        samplers = [
            aniso("clamp-to-edge"),
            aniso("repeat"),
            dev.create_sampler(mag_filter="linear", min_filter="linear"),
        ]
        module = dev.create_shader_module(code=SHADER.read_text())
        self.pipeline = dev.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main"},
            fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": "rgba8unorm"}]},
            primitive={"topology": "triangle-list"},
        )
        resources = [
            {"buffer": self.uniforms},
            self.textures["day"].create_view(dimension="2d-array"),
            self.textures["night"].create_view(),
            self.textures["clouds"].create_view(),
            self.textures["water"].create_view(),
            self.glow.create_view(),
            self.overlay.create_view(),
            *samplers,
        ]
        self.bind_group = dev.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": r} for i, r in enumerate(resources)],
        )

    def _upload_textures(self) -> dict:
        dev, q = self.device, self.device.queue
        out = {}
        # day: 21600 px wide, over the 16384 px limit, so two square halves in an array
        levels = _levels("day")
        w0 = levels[0].shape[1] // 2
        tex = dev.create_texture(
            size=(w0, levels[0].shape[0], 2), format="rgba8unorm", usage=TEX_USAGE, mip_level_count=len(levels)
        )
        for m, lv in enumerate(levels):
            h, w = lv.shape[:2]
            for layer in range(2):
                half = _rgba(lv[:, layer * (w // 2) : (layer + 1) * (w // 2)])
                q.write_texture(
                    {"texture": tex, "mip_level": m, "origin": (0, 0, layer)},
                    half,
                    {"bytes_per_row": (w // 2) * 4, "rows_per_image": h},
                    (w // 2, h, 1),
                )
        out["day"] = tex
        for name, fmt in (("night", "rgba8unorm"), ("clouds", "r8unorm"), ("water", "r8unorm")):
            levels = _levels(name)
            h0, w0 = levels[0].shape[:2]
            tex = dev.create_texture(size=(w0, h0, 1), format=fmt, usage=TEX_USAGE, mip_level_count=len(levels))
            for m, lv in enumerate(levels):
                h, w = lv.shape[:2]
                data = _rgba(lv) if fmt == "rgba8unorm" else np.ascontiguousarray(lv[..., 0] if lv.ndim == 3 else lv)
                bpp = 4 if fmt == "rgba8unorm" else 1
                q.write_texture(
                    {"texture": tex, "mip_level": m}, data, {"bytes_per_row": w * bpp, "rows_per_image": h}, (w, h, 1)
                )
            out[name] = tex
        return out

    def render(self, basis, cam, sun, overlay: bytes, glow: bytes, fade: float, seed: int, tanh: float) -> np.ndarray:
        """Render one frame; returns an (H, W, 4) uint8 array."""
        C, f, r, u = basis
        W, H = self.W, self.H
        S = min(W, H)
        u_data = np.array(
            [
                [*C, cam.D],
                [*f, tanh],
                [*r, cam.sx],
                [*u, cam.sy],
                [*sun, 1.25],
                [W, H, S / 2, 2 * tanh / S],
                [fade, seed % 100000, 0.55 * S / 1080, 1.16],
            ],
            np.float32,
        )
        q = self.device.queue
        q.write_buffer(self.uniforms, 0, u_data)
        q.write_texture({"texture": self.overlay}, overlay, {"bytes_per_row": W * 4, "rows_per_image": H}, (W, H, 1))
        gw, gh = self.glow_size
        q.write_texture({"texture": self.glow}, glow, {"bytes_per_row": gw * 4, "rows_per_image": gh}, (gw, gh, 1))
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {"view": self.target.create_view(), "load_op": "clear", "store_op": "store", "clear_value": (0, 0, 0, 1)}
            ]
        )
        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.bind_group)
        rp.draw(3)
        rp.end()
        q.submit([enc.finish()])
        data = q.read_texture({"texture": self.target}, {"bytes_per_row": W * 4, "rows_per_image": H}, (W, H, 1))
        return np.frombuffer(data, np.uint8).reshape(H, W, 4)


def sun_dir(basis) -> np.ndarray:
    """Sun fixed relative to the camera: upper left, slightly behind."""
    _, f, r, u = basis
    v = -0.62 * r + 0.42 * u + 0.66 * (-f)
    return v / math.sqrt(float(np.dot(v, v)))
