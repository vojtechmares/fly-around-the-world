// One full-screen pass per frame: stars, globe, atmosphere, tone mapping, route
// glow, vignette, the Pillow overlay (routes, labels, HUD), fade and dither.

struct Uniforms {
    cam: vec4f,     // camera position, w: distance from the Earth's centre
    fwd: vec4f,     // forward, w: tan(fov / 2)
    right: vec4f,   // right, w: lens shift x
    up: vec4f,      // up, w: lens shift y
    sun: vec4f,     // direction to the sun, w: exposure
    screen: vec4f,  // width, height, half of the short side, angular size of a pixel
    misc: vec4f,    // fade, dither seed, star sigma (px), halo radius
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var day_tex: texture_2d_array<f32>;  // two halves of the 21600 px map
@group(0) @binding(2) var night_tex: texture_2d<f32>;
@group(0) @binding(3) var cloud_tex: texture_2d<f32>;
@group(0) @binding(4) var water_tex: texture_2d<f32>;
@group(0) @binding(5) var glow_tex: texture_2d<f32>;
@group(0) @binding(6) var overlay_tex: texture_2d<f32>;
@group(0) @binding(7) var clamp_samp: sampler;   // anisotropic, clamped (day halves)
@group(0) @binding(8) var wrap_samp: sampler;    // anisotropic, wraps around in longitude
@group(0) @binding(9) var linear_samp: sampler;  // bilinear, for the glow layer

const PI: f32 = 3.14159265;
const STAR_CELLS: f32 = 48.0;
const STAR_DENSITY: f32 = 0.45;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rnd(seed: u32) -> f32 {
    return f32(pcg(seed)) / 4294967296.0;
}

fn sstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn aces(x: vec3f) -> vec3f {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// Stars fixed to the sky: one jittered candidate per cell of a cube map.
fn stars(d: vec3f) -> vec3f {
    let a = abs(d);
    var face: u32;
    var uv: vec2f;
    if (a.x >= a.y && a.x >= a.z) {
        face = select(1u, 0u, d.x > 0.0);
        uv = d.yz / a.x;
    } else if (a.y >= a.z) {
        face = select(3u, 2u, d.y > 0.0);
        uv = d.xz / a.y;
    } else {
        face = select(5u, 4u, d.z > 0.0);
        uv = d.xy / a.z;
    }
    let cell = clamp(floor((uv * 0.5 + 0.5) * STAR_CELLS), vec2f(0.0), vec2f(STAR_CELLS - 1.0));
    let n = u32(STAR_CELLS);
    let seed = pcg(face * n * n + u32(cell.y) * n + u32(cell.x) + 12345u);
    if (rnd(seed) > STAR_DENSITY) {
        return vec3f(0.0);
    }
    let jitter = vec2f(0.15 + 0.7 * rnd(seed + 1u), 0.15 + 0.7 * rnd(seed + 2u));
    let suv = (cell + jitter) / STAR_CELLS * 2.0 - 1.0;
    let sgn = select(-1.0, 1.0, (face & 1u) == 0u);
    var sd: vec3f;
    if (face < 2u) {
        sd = vec3f(sgn, suv.x, suv.y);
    } else if (face < 4u) {
        sd = vec3f(suv.x, sgn, suv.y);
    } else {
        sd = vec3f(suv.x, suv.y, sgn);
    }
    sd = normalize(sd);
    let dist = length(cross(d, sd)) / U.screen.w;  // in pixels
    let sigma = U.misc.z;
    let inten = 0.9 * (0.04 + 0.9 * pow(rnd(seed + 3u), 12.5));
    let temp = rnd(seed + 4u);
    let col = vec3f(0.85 + 0.15 * temp, 0.9 + 0.05 * temp, 1.0 - 0.2 * temp);
    return col * inten * exp(-dist * dist / (2.0 * sigma * sigma)) * 1.6 / (sigma * sigma + 0.5);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let W = U.screen.x;
    let H = U.screen.y;
    let half_s = U.screen.z;
    let pixang = U.screen.w;
    let C = U.cam.xyz;
    let D = U.cam.w;
    let tanh = U.fwd.w;
    let L = U.sun.xyz;

    // primary ray
    let x = ((pos.x - W / 2.0) / half_s - U.right.w) * tanh;
    let y = (-(pos.y - H / 2.0) / half_s - U.up.w) * tanh;
    let d = normalize(U.fwd.xyz + x * U.right.xyz + y * U.up.xyz);
    let b = dot(d, C);
    let h = sqrt(max(D * D - b * b, 0.0));
    let pw = sqrt(D * D - 1.0) * pixang;
    let cov = clamp((1.0 - h) / pw + 0.5, 0.0, 1.0);

    // surface point (the tangent point for rays that miss, for smooth derivatives)
    let hh = min(h, 1.0);
    let t = -b - sqrt(max(1.0 - hh * hh, 0.0));
    let n = normalize(C + t * d);
    let lon = atan2(-n.z, n.x);
    let lat = asin(clamp(n.y, -1.0, 1.0));
    let uv = vec2f(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI);

    // texture gradients, taken the short way round at the date line
    let uw = fract(uv.x + 0.5);
    let dux0 = dpdx(uv.x);
    let duy0 = dpdy(uv.x);
    let dux1 = dpdx(uw);
    let duy1 = dpdy(uw);
    let dux = select(dux0, dux1, abs(dux1) < abs(dux0));
    let duy = select(duy0, duy1, abs(duy1) < abs(duy0));
    let dvx = dpdx(uv.y);
    let dvy = dpdy(uv.y);
    let gx = vec2f(dux, dvx);
    let gy = vec2f(duy, dvy);

    // the day map is stored as two 10800 px halves (GPU texture size limit)
    let layer = select(0, 1, uv.x >= 0.5);
    let uh = vec2f(uv.x * 2.0 - f32(layer), uv.y);
    let gs = vec2f(2.0, 1.0);
    let day = pow(textureSampleGrad(day_tex, clamp_samp, uh, layer, gx * gs, gy * gs).rgb, vec3f(2.2));
    let night = pow(textureSampleGrad(night_tex, wrap_samp, uv, gx, gy).rgb, vec3f(2.2));
    let cloud = textureSampleGrad(cloud_tex, wrap_samp, uv, gx, gy).r;
    let water = textureSampleGrad(water_tex, wrap_samp, uv, gx, gy).r;

    // surface shading
    let V = -d;
    let ndl = dot(n, L);
    let diff = clamp((ndl + 0.08) / 1.08, 0.0, 1.0);
    let dayk = sstep(-0.12, 0.22, ndl);
    let c = pow(clamp(cloud * 1.1 - 0.1, 0.0, 1.0), 1.4) * 0.6;
    var surf = day * (0.003 + 1.3 * diff) * vec3f(1.0, 0.97, 0.92);
    let cloud_col = (0.015 + 0.85 * diff) * vec3f(0.95, 0.96, 1.0);
    surf = surf * (1.0 - c) + cloud_col * c;
    surf += night * vec3f(1.0, 0.82, 0.55) * 2.2 * ((1.0 - dayk) * (1.0 - 0.7 * c));
    let Hv = normalize(L + V);
    let ndh = clamp(dot(n, Hv), 0.0, 1.0);
    let lit = select(0.0, 1.0, ndl > 0.0);
    let spec = (pow(ndh, 120.0) * 0.6 + pow(ndh, 20.0) * 0.015) * water * (1.0 - c) * lit * diff;
    surf += spec * vec3f(1.0, 0.92, 0.8);

    // atmosphere
    let ra = 1.03;
    let ra_halo = U.misc.w;
    let path_out = 2.0 * sqrt(max(ra * ra - h * h, 0.0));
    let path_in = sqrt(max(ra * ra - h * h, 0.0)) - sqrt(max(1.0 - h * h, 0.0));
    let path = cov * path_in * 0.35 + (1.0 - cov) * path_out;
    let ncl = C - b * d;
    let nc = ncl / max(length(ncl), 1e-6);
    var cosl = dot(nc, L);
    cosl = cov * ndl + (1.0 - cov) * cosl;
    var lf = sstep(-0.22, 0.25, cosl);
    lf *= (1.0 - cov) + cov * sstep(-0.05, 0.3, cosl);
    let band = exp(-pow((cosl - 0.03) / 0.12, 2.0)) * (0.45 - 0.3 * cov);
    let sky = vec3f(0.30, 0.58, 1.0);
    let sunset = vec3f(1.0, 0.55, 0.25);
    let acol = sky + (sunset - sky) * band;
    var atmo = acol * ((1.0 - exp(-path * 7.0)) * lf * 1.2);
    let halo = exp(-max(h - 1.0, 0.0) / 0.045) * (1.0 - cov) * lf * 0.2 * clamp((ra_halo - 0.01 - h) / 0.08, 0.0, 1.0);
    atmo += sky * halo;

    let bg = stars(d);
    let lin = bg * (1.0 - cov) + surf * cov + atmo;

    // to display, then the 2D layers
    var col = pow(aces(lin * U.sun.w), vec3f(1.0 / 2.2));
    col += textureSample(glow_tex, linear_samp, pos.xy / vec2f(W, H)).rgb * 0.9;
    let rx = (pos.x - W / 2.0) / (W / 2.0);
    let ry = (pos.y - H / 2.0) / (H / 2.0);
    col *= 1.0 - 0.22 * pow(clamp((rx * rx + ry * ry) / 2.0, 0.0, 1.0), 1.5);
    let ov = textureLoad(overlay_tex, vec2i(pos.xy), 0);
    col = col * (1.0 - ov.a) + ov.rgb * ov.a;
    col *= U.misc.x;
    let ip = vec2u(pos.xy);
    let noise = rnd(ip.x * 7919u + ip.y * 104729u + u32(U.misc.y) * 15485863u);
    return vec4f(clamp(col + (noise - 0.5) / 255.0, vec3f(0.0), vec3f(1.0)), 1.0);
}
