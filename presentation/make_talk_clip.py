"""Talk version of the marked Para-Lane clip (slide 7): the first 12 s of results/videos/paralane_straight_3m_scene16.mp4
without its title and legend bars (the slide's title and caption state both), with the six panel labels redrawn at
24 px so that they stay legible when projected. Writes img/talk_clip_3m.mp4 and img/talk_clip_3m_poster.jpg (the
frame with the most spurious-vehicle marks, so that the still slide shows every mark the caption names).
Needs ffmpeg. usage (from this folder): python make_talk_clip.py"""
import glob
import os
import subprocess
import tempfile

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "videos", "paralane_straight_3m_scene16.mp4")
BOLD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")  # the font of src/viz/compose.py
# layout of the source frames (src/viz/compose.py: frame_image, 3 columns, tile 426 x 240, 2 px gaps, 50 px title bar,
# 24 px legend bar): tile origins after cropping the title bar
TOP, H = 50, 2 * 240 + 2
X, Y = (-1, 427, 855), (0, 242)
OLD = ["real frame, lane 1", "raw splat", "released Difix", "our Difix (reimpl., ref)", "our small U-Net", "NVIDIA Fixer"]
NEW = ["real photo", "raw render", "released Difix", "our re-implementation", "our small U-Net", "NVIDIA Fixer"]


def relabel(im):
    im = im.crop((0, TOP, im.width, TOP + H)); d = ImageDraw.Draw(im)
    f_old, f_new = ImageFont.truetype(BOLD, 15), ImageFont.truetype(BOLD, 24)
    for k, (old, new) in enumerate(zip(OLD, NEW)):
        x, y = X[k % 3], Y[k // 3]
        w = max(d.textlength(new, font=f_new) + 18, d.textlength(old, font=f_old) + 12)  # cover the old label fully
        d.rectangle([x, y, x + w, y + 34], fill=(16, 16, 20))
        d.text((x + 9, y + 3), new, fill=(255, 255, 255), font=f_new)
    return im


def main():
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["ffmpeg", "-v", "error", "-i", SRC, "-t", "12", f"{t}/f_%03d.png"], check=True)
        fs = sorted(glob.glob(f"{t}/f_*.png"))
        red = []
        for p in fs:
            im = relabel(Image.open(p).convert("RGB")); im.save(p)
            a = np.asarray(im, np.int16)  # spurious-vehicle boxes are drawn in (230, 30, 30) (src/viz/annotate.py)
            red.append(int(((a[..., 0] > 190) & (a[..., 1] < 70) & (a[..., 2] < 70)).sum()))
        Image.open(fs[int(np.argmax(red))]).save(os.path.join(HERE, "img", "talk_clip_3m_poster.jpg"), quality=90)
        print("poster: frame", int(np.argmax(red)), "of", len(fs), "with", max(red), "red-mark pixels")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "5", "-i", f"{t}/f_%03d.png", "-c:v", "libx264",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                        os.path.join(HERE, "img", "talk_clip_3m.mp4")], check=True)
    print(f"{len(fs)} frames, 1280 x {H} -> img/talk_clip_3m.mp4 and img/talk_clip_3m_poster.jpg")


if __name__ == "__main__":
    main()
