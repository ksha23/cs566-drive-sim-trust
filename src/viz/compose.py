"""Side-by-side demo videos for the project webpage: a labeled grid of tiles (real | raw | fixers ...), a header with
title and a per-frame status line (dataset, scenario, offset), a footer with the dataset attribution. H.264 (libx264,
yuv420p, faststart), 1280 px wide, re-encoded at a higher CRF until the file is <= 9.5 MB.

API: write_video(out_mp4, frames, title, attribution, cols=3, fps=5, tile_w=426)
  frames: list of dict(tiles=[(label, image_path_or_array_or_None, missing_text), ...], status="...", note="...")
  A tile whose image is None is drawn dark with its missing_text (e.g. "no real frame at this offset").
still(out_jpg, frame, ...) writes one frame as a JPEG (<= 500 KB) for the site header.
"""
import os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_f = {}


def font(sz, bold=False):
    k = (sz, bold)
    if k not in _f:
        _f[k] = ImageFont.truetype(FONTB if bold else FONT, sz)
    return _f[k]


def _img(x, size):
    if x is None:
        return None
    im = Image.open(x).convert("RGB") if isinstance(x, str) else Image.fromarray(np.asarray(x, np.uint8))
    return im.resize(size, Image.LANCZOS)


def tile(label, src, missing, size, sub=None):
    im = _img(src, size)
    if im is None:
        im = Image.new("RGB", size, (38, 38, 42)); d = ImageDraw.Draw(im)
        lines = (missing or "no image").split("\n")
        for k, l in enumerate(lines):
            w = d.textlength(l, font=font(15))
            d.text(((size[0] - w) / 2, size[1] / 2 - 10 * len(lines) + 20 * k), l, fill=(200, 200, 200), font=font(15))
    d = ImageDraw.Draw(im, "RGBA")
    w = d.textlength(label, font=font(15, True))
    d.rectangle([0, 0, w + 12, 22], fill=(0, 0, 0, 170)); d.text((6, 3), label, fill=(255, 255, 255), font=font(15, True))
    if sub:
        w = d.textlength(sub, font=font(13))
        d.rectangle([0, size[1] - 20, w + 10, size[1]], fill=(0, 0, 0, 160))
        d.text((5, size[1] - 18), sub, fill=(255, 235, 150), font=font(13))
    return im


def frame_image(fr, title, attribution, cols=3, tile_w=426, total_w=1280):
    size = (tile_w, int(round(tile_w * 9 / 16 / 2)) * 2)
    tiles = [tile(t[0], t[1], t[2] if len(t) > 2 else None, size, t[3] if len(t) > 3 else None) for t in fr["tiles"]]
    rows = (len(tiles) + cols - 1) // cols
    head, foot = 50, 24
    Hh = head + rows * size[1] + foot + (rows - 1) * 2
    Hh += Hh % 2
    canvas = Image.new("RGB", (total_w, Hh), (16, 16, 20)); d = ImageDraw.Draw(canvas)
    x0 = (total_w - cols * size[0] - (cols - 1) * 2) // 2
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas.paste(t, (x0 + c * (size[0] + 2), head + r * (size[1] + 2)))
    if len(tiles) < rows * cols and fr.get("note"):  # free slot: explanatory text
        k = len(tiles); r, c = divmod(k, cols); bx, by = x0 + c * (size[0] + 2), head + r * (size[1] + 2)
        d.rectangle([bx, by, bx + size[0], by + size[1]], fill=(28, 28, 34))
        _wrap(d, fr["note"], bx + 10, by + 10, size[0] - 20, font(14), (220, 220, 220))
    d.text((10, 5), title, fill=(255, 255, 255), font=font(18, True))
    d.text((10, 28), fr.get("status", ""), fill=(255, 220, 120), font=font(15))
    d.text((10, Hh - foot + 4), attribution, fill=(160, 160, 170), font=font(12))
    if fr.get("legend"):  # [(rgb, label), ...] right-aligned in the footer
        x = total_w - 10
        for col, lab in reversed(fr["legend"]):
            x -= d.textlength(lab, font=font(12)); d.text((x, Hh - foot + 4), lab, fill=(220, 220, 220), font=font(12))
            x -= 16; d.rectangle([x, Hh - foot + 6, x + 11, Hh - foot + 17], fill=col); x -= 14
    return canvas


def _wrap(d, text, x, y, w, f, fill):
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            t = (line + " " + word).strip()
            if d.textlength(t, font=f) > w and line:
                d.text((x, y), line, fill=fill, font=f); y += f.size + 4; line = word
            else:
                line = t
        d.text((x, y), line, fill=fill, font=f); y += f.size + 8


def write_video(out, frames, title, attribution, cols=3, fps=5, tile_w=426, max_mb=4.5, max_s=20.0):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    frames = frames[: int(max_s * fps)]
    imgs = [np.asarray(frame_image(f, title, attribution, cols, tile_w)) for f in frames]
    Hh, Ww = imgs[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for crf in (23, 26, 29, 32, 35):
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Ww}x{Hh}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for im in imgs:
            p.stdin.write(np.ascontiguousarray(im).tobytes())
        p.stdin.close(); p.wait()
        mb = os.path.getsize(out) / 2 ** 20
        if mb <= max_mb:
            break
    print(f"wrote {out}: {len(imgs)} frames @ {fps} fps = {len(imgs)/fps:.1f} s, {Ww}x{Hh}, {mb:.2f} MB (crf {crf})", flush=True)
    return dict(frames=len(imgs), fps=fps, seconds=len(imgs) / fps, width=Ww, height=Hh, mb=round(mb, 2), crf=crf)


def still(out, fr, title, attribution, cols=3, tile_w=426, max_kb=480):
    im = frame_image(fr, title, attribution, cols, tile_w)
    for q in (88, 82, 76, 70, 60, 50):
        im.save(out, quality=q, optimize=True, progressive=True)
        if os.path.getsize(out) / 1024 <= max_kb:
            break
    print(f"wrote {out}: {os.path.getsize(out)/1024:.0f} KB", flush=True)
