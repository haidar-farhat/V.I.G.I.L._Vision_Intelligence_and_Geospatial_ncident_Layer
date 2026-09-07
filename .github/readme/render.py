"""Render the README's figures as SVG with every letter as an outline.

GitHub shows a README's images through a proxy, as plain <img> elements, so an
SVG there cannot load a web font: whatever `font-family` it names, the viewer
gets whatever their machine has. Converting the text to glyph outlines with
HarfBuzz (shaping and kerning) and fontTools (the outlines) makes the figure
render identically everywhere, with no font dependency at all.

Two files per figure — a dark and a light one — because an image cannot read
the page's theme; the README chooses between them with <picture>.

    python .github/readme/render.py

Fonts: Barlow Condensed and IBM Plex Mono, both OFL, read from
%LOCALAPPDATA%/vigil-build/fonts (or $FONT_DIR). Fetch them from the
google/fonts repository under ofl/barlowcondensed and ofl/ibmplexmono.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
FONT_DIR = Path(os.environ.get("FONT_DIR") or Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "vigil-build" / "fonts")

FACES = {
    "display-bold": "BarlowCondensed-Bold.ttf",
    "display-semi": "BarlowCondensed-SemiBold.ttf",
    "display-medium": "BarlowCondensed-Medium.ttf",
    "mono": "IBMPlexMono-Regular.ttf",
    "mono-medium": "IBMPlexMono-Medium.ttf",
}

THEMES = {
    # GitHub's dark ground is #0d1117; the figures sit on it with no background of their own.
    "dark": dict(ink="#e4eaef", ink2="#99a6b3", ink3="#6b7885", line="#23303e", line2="#34445a",
                 panel="#121923", panel2="#19222d", stake="#e8743b", stake_soft="#e8743b29",
                 survey="#7fb2e5", survey_soft="#7fb2e529", grid="#e4eaef22"),
    "light": dict(ink="#141a21", ink2="#57636f", ink3="#8a96a2", line="#d3dae1", line2="#b6c0ca",
                  panel="#f3f5f7", panel2="#e8ecf0", stake="#d4622b", stake_soft="#d4622b1f",
                  survey="#2f6fb0", survey_soft="#2f6fb024", grid="#141a2126"),
}


class Face:
    def __init__(self, path: Path):
        self.tt = TTFont(str(path))
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upem = self.tt["head"].unitsPerEm
        blob = hb.Blob.from_file_path(str(path))
        self.hb = hb.Font(hb.Face(blob))
        self.hb.scale = (self.upem, self.upem)

    def shape(self, text: str):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        return list(zip(buf.glyph_infos, buf.glyph_positions))

    def width(self, text: str, size: float) -> float:
        return sum(pos.x_advance for _, pos in self.shape(text)) * size / self.upem


_faces: dict[str, Face] = {}


def face(name: str) -> Face:
    if name not in _faces:
        _faces[name] = Face(FONT_DIR / FACES[name])
    return _faces[name]


def text(x: float, y: float, string: str, *, face_name: str, size: float, fill: str,
         anchor: str = "start", tracking: float = 0.0, opacity: float | None = None) -> str:
    """One <path> per string: the text as outlines, positioned like SVG text would be."""
    f = face(face_name)
    scale = size / f.upem
    track = tracking * size
    width = f.width(string, size) + track * max(len(string) - 1, 0)
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    d = []
    pen_x = x
    for info, pos in f.shape(string):
        # One decimal at these sizes is invisible and a fifth of the file.
        pen = SVGPathPen(f.glyphs, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip(".") or "0")
        # y up in the font, y down on the page: flip, then place.
        transform = (scale, 0, 0, -scale, pen_x + pos.x_offset * scale, y - pos.y_offset * scale)
        f.glyphs[f.order[info.codepoint]].draw(TransformPen(pen, transform))
        d.append(pen.getCommands())
        pen_x += pos.x_advance * scale + track
    extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return f'<path fill="{fill}"{extra} d="{" ".join(p for p in d if p)}"/>'


# ------------------------------------------------------------------ the hero

def hero(t: dict) -> str:
    W, H = 1200, 440
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'aria-label="Sentinel Vision: it says what it measured. A site plan drawn from two cameras, a tracked person '
         'with an error ellipse 11.2 plus or minus 2.4 metres from a restricted zone, and a car with three people apparently inside.">']
    # -- left column
    x = 8
    o.append(text(x, 86, "SENTINEL VISION V2  ·  LOCAL-FIRST", face_name="display-medium", size=15, fill=t["stake"], tracking=0.16))
    o.append(text(x - 3, 178, "It says what", face_name="display-bold", size=104, fill=t["ink"]))
    o.append(text(x - 3, 274, "it ", face_name="display-bold", size=104, fill=t["ink"]))
    it_w = face("display-bold").width("it ", 104)
    o.append(text(x - 3 + it_w, 274, "measured.", face_name="display-bold", size=104, fill=t["stake"]))
    dek = ["Ordinary cameras in. A small number of reviewable",
           "incidents out, each with the evidence that produced it.",
           "No route to the Internet at any point."]
    for i, line in enumerate(dek):
        o.append(text(x, 318 + i * 27, line, face_name="display-medium", size=22, fill=t["ink2"]))
    facts = [("454 + 77", "tests"), ("42", "capabilities tested"), ("0", "outbound connections")]
    fx = x
    for n, label in facts:
        o.append(text(fx, 420, n, face_name="mono-medium", size=14, fill=t["ink"]))
        fx += face("mono-medium").width(n, 14) + 6
        o.append(text(fx, 420, label, face_name="mono", size=14, fill=t["ink2"]))
        fx += face("mono").width(label, 14) + 22

    # -- right: the plan, in a panel
    px, py, pw, ph = 612, 22, 580, 400
    o.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4" fill="{t["panel"]}" stroke="{t["line"]}"/>')
    o.append(f'<clipPath id="plan"><rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="4"/></clipPath>')
    o.append(f'<g clip-path="url(#plan)">')
    grid = []
    for gx in range(px, px + pw + 1, 40):
        grid.append(f"M{gx} {py}V{py + ph}")
    for gy in range(py, py + ph + 1, 40):
        grid.append(f"M{px} {gy}H{px + pw}")
    o.append(f'<path d="{" ".join(grid)}" stroke="{t["grid"]}" stroke-width="1" fill="none"/>')
    o.append('</g>')
    # plan coordinates: same drawing as the site's, translated into the panel
    o.append(f'<g clip-path="url(#plan)"><g transform="translate({px + 20} {py + 4}) scale(0.9)">')
    o.append(f'<polygon points="60,380 250,20 560,200" fill="{t["survey_soft"]}" stroke="{t["survey"]}"/>')
    o.append(f'<polygon points="560,400 160,100 80,330" fill="{t["survey_soft"]}" stroke="{t["survey"]}"/>')
    o.append(f'<rect x="56" y="376" width="8" height="8" transform="rotate(45 60 380)" fill="{t["ink"]}"/>')
    o.append(f'<rect x="556" y="396" width="8" height="8" transform="rotate(45 560 400)" fill="{t["ink"]}"/>')
    o.append(text(72, 404, "north-gate", face_name="mono-medium", size=11, fill=t["ink"]))
    o.append(text(72, 418, "4 m mast · heading ±0.06° measured", face_name="mono", size=11, fill=t["ink2"]))
    o.append(text(548, 424, "yard-east", face_name="mono-medium", size=11, fill=t["ink"], anchor="end"))
    o.append(text(548, 436, "8 m · ±2° assumed, not measured", face_name="mono", size=11, fill=t["ink2"], anchor="end"))
    o.append(f'<polygon points="250,150 470,130 500,290 280,320" fill="{t["stake_soft"]}" stroke="{t["stake"]}" '
             f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    o.append(text(266, 172, "YARD", face_name="display-semi", size=14, fill=t["stake"], tracking=0.14))
    o.append(text(266, 187, "restricted 22–06", face_name="mono", size=11, fill=t["ink2"]))
    o.append(f'<ellipse cx="412" cy="214" rx="34" ry="15" transform="rotate(49.5 412 214)" fill="{t["survey_soft"]}" '
             f'stroke="{t["survey"]}" stroke-width="1.2"/>')
    o.append(f'<rect x="388" y="202" width="48" height="24" rx="3" fill="none" stroke="{t["ink"]}" stroke-width="1.4"/>')
    o.append(text(412, 250, "car · 0.91", face_name="mono-medium", size=11, fill=t["ink"], anchor="middle"))
    o.append(text(412, 264, "probably 3 inside", face_name="mono", size=11, fill=t["ink2"], anchor="middle"))
    o.append(f'<ellipse cx="215" cy="255" rx="26" ry="11" transform="rotate(-39 215 255)" fill="{t["survey_soft"]}" '
             f'stroke="{t["survey"]}" stroke-width="1.2"/>')
    o.append(f'<circle cx="215" cy="255" r="4" fill="{t["stake"]}"/>')
    o.append(f'<line x1="221" y1="252" x2="252" y2="238" stroke="{t["ink"]}" stroke-width="1.4"/>'
             f'<polygon points="256,236 247,236.5 250.5,243" fill="{t["ink"]}"/>')
    o.append(text(150, 236, "person · 0.84", face_name="mono-medium", size=11, fill=t["ink"]))
    o.append(text(150, 250, "approaching", face_name="mono", size=11, fill=t["ink2"]))
    o.append(f'<line x1="219" y1="256" x2="268" y2="256" stroke="{t["stake"]}" stroke-dasharray="2 3"/>')
    o.append(text(243, 278, "11.2 ± 2.4 m", face_name="mono-medium", size=13, fill=t["stake"], anchor="middle"))
    o.append(text(14, 24, "□ = 5 m · drawn from the cameras' own geometry, no tiles", face_name="mono", size=11, fill=t["ink3"]))
    o.append('</g></g>')
    o.append('</svg>')
    return "\n".join(o)


# -------------------------------------------------------------- the pipeline

STAGES = [
    ("Decode", "public addresses", "refused"),
    ("Frame quality", "blur, glare, freeze", "named, not silent"),
    ("Camera motion", "a passing lorry", "is not the camera"),
    ("Detect", "any ONNX model;", "NMS per class"),
    ("Describe", "an occluded crop", "is discarded"),
    ("Track", "Kalman; optimal", "assignment"),
    ("Project", "a ray at the horizon", "is TOO_SHALLOW"),
    ("Zones & rules", "hysteresis; warns", "before the breach"),
    ("Correlate", "three cameras,", "one incident"),
    ("Review", "a dismissal needs", "a written reason"),
]
CARRIES = ["frames", "usable frames", "warped filters", "boxes + masks", "descriptors", "tracks", "position ± error", "events", "incidents"]


def pipeline(t: dict) -> str:
    W, H = 1100, 236
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'aria-label="The pipeline from decode to review: what each stage carries forward and what it refuses. '
         'Stages one to seven report; zones, correlation and review make a claim.">']
    for i, (name, r1, r2) in enumerate(STAGES):
        x = 8 + i * 109
        last = i == len(STAGES) - 1
        fill = t["stake_soft"] if last else t["panel2"]
        stroke = t["stake"] if last else t["line2"]
        o.append(f'<rect x="{x}" y="64" width="92" height="46" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="{1.5 if last else 1}"/>')
        o.append(text(x + 46, 92, name, face_name="display-semi", size=16, fill=t["ink"], anchor="middle"))
        o.append(text(x + 46, 132, r1, face_name="display-medium", size=13, fill=t["ink2"], anchor="middle"))
        o.append(text(x + 46, 147, r2, face_name="display-medium", size=13, fill=t["ink2"], anchor="middle"))
        if i < len(CARRIES):
            ax = x + 100
            o.append(f'<line x1="{ax}" y1="87" x2="{ax + 9}" y2="87" stroke="{t["ink2"]}" stroke-width="1.2"/>'
                     f'<polygon points="{ax + 15},87 {ax + 9},84 {ax + 9},90" fill="{t["ink2"]}"/>')
            o.append(text(ax + 8, 54, CARRIES[i], face_name="mono", size=10.5, fill=t["survey"], anchor="middle"))
    o.append(f'<path d="M8 178v8h746v-8" fill="none" stroke="{t["stake"]}"/>')
    o.append(text(381, 208, "REPORTS", face_name="display-semi", size=13, fill=t["stake"], anchor="middle", tracking=0.14))
    o.append(f'<path d="M771 178v8h310v-8" fill="none" stroke="{t["stake"]}"/>')
    o.append(text(926, 208, "MAKES A CLAIM · MUST JUSTIFY IT", face_name="display-semi", size=13, fill=t["stake"], anchor="middle", tracking=0.14))
    o.append('</svg>')
    return "\n".join(o)


def main() -> int:
    missing = [n for n in FACES.values() if not (FONT_DIR / n).is_file()]
    if missing:
        print(f"fonts missing from {FONT_DIR}: {missing}")
        return 1
    for theme, tokens in THEMES.items():
        for name, draw in (("hero", hero), ("pipeline", pipeline)):
            out = HERE / f"{name}-{theme}.svg"
            out.write_text(draw(tokens), encoding="utf-8")
            print(f"wrote {out.relative_to(HERE.parent.parent)} ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
