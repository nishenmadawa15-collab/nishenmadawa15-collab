"""
Generate a terminal/ASCII-style SVG portrait from nishen-profile.jpeg.

Pipeline:
1. Load the source photo and separate subject from the studio backdrop
   using border-connected flood fill on a brightness-difference mask
   (the backdrop is a smooth, roughly uniform gray; the subject is not).
2. Downsample the subject region to a character grid, sized so that
   grid_cols/grid_rows * cell_w/cell_h reproduces the source aspect ratio.
3. Map per-cell luminance to a density ramp of ASCII glyphs and a
   cyan -> white color ramp; cells outside the subject mask render as
   faint "code rain" glyphs or empty space instead of the portrait ramp.
4. Emit the grid as monospace <text>/<tspan> runs inside an SVG: a
   rounded, softly glowing panel with a thin accent border (no terminal
   window chrome) so it reads as a clean portrait, not a hacker mockup.

Usage: python generate_ascii_portrait.py
Reads:  ../nishen-profile.jpeg (relative to this script)
Writes: ../assets/nishen-ascii.svg
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGE = ROOT / "nishen-profile.jpeg"
OUT_SVG = ROOT / "assets" / "nishen-ascii.svg"

COLS = 108
CELL_W = 3.9
CELL_H = 7.8
PADDING = 16

RAMP = ".'`,:;-~+=iltI?!*xnuvcz1XY0OQ#%$@"
CODE_GLYPHS = "01{}<>/;:.="

COLOR_STOPS = [
    (0.00, (18, 60, 85)),
    (0.18, (12, 90, 120)),
    (0.38, (0, 140, 170)),
    (0.60, (44, 195, 210)),
    (0.80, (150, 232, 236)),
    (1.00, (235, 250, 250)),
]


def lerp(a, b, t):
    return a + (b - a) * t


def ramp_color(t):
    t = max(0.0, min(1.0, t))
    for i in range(len(COLOR_STOPS) - 1):
        t0, c0 = COLOR_STOPS[i]
        t1, c1 = COLOR_STOPS[i + 1]
        if t0 <= t <= t1:
            local = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            r = round(lerp(c0[0], c1[0], local))
            g = round(lerp(c0[1], c1[1], local))
            b = round(lerp(c0[2], c1[2], local))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ffffff"


def subject_mask(gray, sat):
    """Border-connected flood fill on a background-likeness mask.

    Erodes the candidate background mask before labeling so thin,
    low-contrast bridges (e.g. jaw/neck shadow) can't leak the flood
    fill into the face, then reconstructs the full region by
    propagating back out through the un-eroded mask.
    """
    # Tight bust-crop portraits fill the bottom corners with the
    # subject's shoulders, so only the top strip and the upper side
    # margins are reliably backdrop.
    h, w = gray.shape
    band = 24
    upper = int(h * 0.4)
    samples = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[:upper, :band].ravel(),
            gray[:upper, -band:].ravel(),
        ]
    )
    bg_level = np.median(samples)

    smooth = ndimage.uniform_filter(gray, size=9)
    local_var = ndimage.uniform_filter(gray**2, size=9) - smooth**2
    local_std = np.sqrt(np.clip(local_var, 0, None))

    # Neutral-gray backdrop: close to backdrop brightness, low local
    # contrast, and low color saturation (skin is visibly warmer/more
    # saturated than the gray studio backdrop even at similar brightness).
    bg_like = (np.abs(gray - bg_level) < 32) & (local_std < 10) & (sat < 0.18)

    seed_radius = 5
    # border_value=1 keeps genuine backdrop touching the frame edge from
    # being eroded away (default border_value=0 treats the area outside
    # the array as background, which would erase real border background).
    eroded = ndimage.binary_erosion(
        bg_like, structure=np.ones((seed_radius, seed_radius)), border_value=1
    )
    labels, _ = ndimage.label(eroded)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)
    seed = np.isin(labels, list(border_labels))

    background = ndimage.binary_propagation(seed, mask=bg_like)
    background = ndimage.binary_dilation(background, structure=np.ones((3, 3)))
    subject = ~background

    subject = ndimage.binary_closing(subject, structure=np.ones((7, 7)), iterations=2)
    subject = ndimage.binary_fill_holes(subject)
    subject = ndimage.binary_opening(subject, structure=np.ones((3, 3)))

    # Discard any small stray subject islands (noise), keep the largest
    # connected component plus anything touching it after closing.
    comp, n = ndimage.label(subject)
    if n > 1:
        sizes = ndimage.sum(subject, comp, range(1, n + 1))
        biggest = 1 + int(np.argmax(sizes))
        subject = comp == biggest

    return subject


def build():
    img = Image.open(SRC_IMAGE).convert("RGB")
    src_w, src_h = img.size

    rows = round(COLS * CELL_W / CELL_H * (src_h / src_w))

    crop_aspect = (COLS * CELL_W) / (rows * CELL_H)
    if src_w / src_h > crop_aspect:
        new_w = round(src_h * crop_aspect)
        x0 = (src_w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, src_h))
    else:
        new_h = round(src_w / crop_aspect)
        y0 = max(0, (src_h - new_h) // 3)
        img = img.crop((0, y0, src_w, min(src_h, y0 + new_h)))

    gray_full = np.array(img.convert("L"), dtype=np.float32)
    hsv_full = np.array(img.convert("HSV"), dtype=np.float32) / 255.0
    sat_full = hsv_full[:, :, 1]
    mask_full = subject_mask(gray_full, sat_full)

    small_img = img.resize((COLS, rows), Image.LANCZOS)
    gray_small = np.array(small_img.convert("L"), dtype=np.float32) / 255.0

    mask_img = Image.fromarray((mask_full * 255).astype(np.uint8)).resize((COLS, rows), Image.LANCZOS)
    mask_small = np.array(mask_img) > 127

    p2, p98 = np.percentile(gray_small[mask_small], (2, 98)) if mask_small.any() else (0, 1)
    gray_norm = np.clip((gray_small - p2) / max(p98 - p2, 1e-4), 0, 1)
    gray_norm = gray_norm**0.85

    rng = random.Random(42)
    width_px = COLS * CELL_W + PADDING * 2
    height_px = rows * CELL_H + PADDING * 2

    rows_svg = []
    for ry in range(rows):
        runs = []
        cur_color = None
        cur_chars = []
        for rx in range(COLS):
            if mask_small[ry, rx]:
                t = float(gray_norm[ry, rx])
                idx = min(len(RAMP) - 1, int(t * (len(RAMP) - 1)))
                ch = RAMP[idx]
                color = ramp_color(t)
            else:
                if rng.random() < 0.02:
                    ch = rng.choice(CODE_GLYPHS)
                    color = "#0b2530"
                else:
                    ch = " "
                    color = "#0b2530"
            if ch == " ":
                if cur_chars and cur_chars[-1] != " ":
                    runs.append((cur_color, "".join(cur_chars)))
                    cur_chars = []
                cur_chars.append(" ")
                cur_color = color
                continue
            if color != cur_color:
                if cur_chars:
                    runs.append((cur_color, "".join(cur_chars)))
                cur_chars = [ch]
                cur_color = color
            else:
                cur_chars.append(ch)
        if cur_chars:
            runs.append((cur_color, "".join(cur_chars)))

        y = PADDING + ry * CELL_H + CELL_H * 0.82
        tspans = "".join(
            f'<tspan fill="{color}">{escape_xml(chars)}</tspan>'
            for color, chars in runs
            if chars.strip() != "" or True
        )
        rows_svg.append(f'<text x="{PADDING}" y="{y:.2f}" xml:space="preserve">{tspans}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.0f}" height="{height_px:.0f}"
     viewBox="0 0 {width_px:.0f} {height_px:.0f}" role="img"
     aria-label="ASCII-style portrait of Nishen Madava">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0a141f"/>
      <stop offset="1" stop-color="#060b12"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="0.5" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <style>
      text {{ font-family: 'Cascadia Code','Fira Code',Consolas,'SFMono-Regular',Menlo,monospace;
              font-size: {CELL_H * 0.92:.2f}px; letter-spacing: 0; }}
    </style>
  </defs>

  <rect x="0.75" y="0.75" width="{width_px - 1.5:.1f}" height="{height_px - 1.5:.1f}" rx="16" fill="url(#bg)" stroke="#00ff88" stroke-opacity="0.3" stroke-width="1.5"/>

  <g filter="url(#glow)">
{chr(10).join(rows_svg)}
  </g>
</svg>
"""
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT_SVG} ({width_px:.0f}x{height_px:.0f}, {COLS}x{rows} grid)")


def escape_xml(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    build()
