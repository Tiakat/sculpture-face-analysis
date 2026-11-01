"""Generate the application icon (PNG + ICO) — a neutral globe with meridians and parallels (no face)."""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def draw(size: int, bg=(27, 42, 56), sea=(38, 84, 124), line=(200, 224, 240), land=(122, 176, 120)) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 256
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(48 * s), fill=bg)
    cx, cy, r = 128 * s, 128 * s, 92 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=sea, outline=line, width=int(4 * s))
    # stylised continents
    d.polygon([(90 * s, 70 * s), (120 * s, 60 * s), (140 * s, 80 * s), (125 * s, 110 * s), (100 * s, 118 * s), (82 * s, 96 * s)], fill=land)
    d.polygon([(104 * s, 126 * s), (128 * s, 130 * s), (136 * s, 160 * s), (120 * s, 196 * s), (104 * s, 170 * s), (96 * s, 146 * s)], fill=land)
    d.polygon([(150 * s, 74 * s), (196 * s, 70 * s), (214 * s, 100 * s), (196 * s, 128 * s), (166 * s, 122 * s), (148 * s, 100 * s)], fill=land)
    d.polygon([(168 * s, 150 * s), (192 * s, 146 * s), (204 * s, 168 * s), (186 * s, 186 * s), (168 * s, 176 * s)], fill=land)
    # meridians & parallels
    for k in (0.35, 0.7):
        d.ellipse([cx - r * k, cy - r, cx + r * k, cy + r], outline=line, width=int(2 * s))
    d.line([cx, cy - r, cx, cy + r], fill=line, width=int(2 * s))
    for k in (-0.5, 0.0, 0.5):
        y = cy + r * k
        half = (r ** 2 - (r * k) ** 2) ** 0.5
        d.line([cx - half, y, cx + half, y], fill=line, width=int(2 * s))
    return im


def main():
    png = draw(512)
    png.save(HERE / "icon.png")
    png.save(HERE / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon.png / icon.ico written")


if __name__ == "__main__":
    main()
