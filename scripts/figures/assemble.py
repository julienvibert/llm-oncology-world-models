"""Assemble figure panels into figures, from scripts/figures/layouts.json.

Each panel was saved at its printed size without a letter. The layout gives, per figure,
the panel ids and their top-left positions in mm; letters are added here (8 pt bold,
lowercase, Nature style). Rearranging figures means editing layouts.json only.

Run:  uv run python scripts/figures/assemble.py            # all figures
      uv run python scripts/figures/assemble.py "Figure 2"  # one figure
      uv run python scripts/figures/assemble.py --allow-missing  # skip figures with missing panels

A figure with a missing panel is an error: the script lists every missing panel and exits 1
without assembling anything. With --allow-missing, those figures are skipped, the rest are assembled and the exit
code is 0 (for partial checkouts).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ASSEMBLED, PANELS  # noqa: E402

MM_TO_PT = 72 / 25.4
LABEL_PT = 8
LABEL_BAND_MM = 3.5  # band reserved above each row of panels for the panel letters


def _panel_size_mm(panel_id: str) -> tuple[float, float]:
    meta = json.loads((PANELS / panel_id / f"{panel_id}_meta.json").read_text())
    return meta["width_mm"], meta["height_mm"]


def _nested(svg_path: Path, x_mm: float, y_mm: float, prefix: str) -> str:
    raw = svg_path.read_text()
    raw = re.sub(r"<\?xml[^>]*>", "", raw)
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw)
    raw = re.sub(r"<metadata>.*?</metadata>", "", raw, flags=re.S)
    raw = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', raw)
    raw = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", raw)
    raw = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}{m.group(1)}"', raw)
    head, rest = raw.split("<svg ", 1)
    tag, body = rest.split(">", 1)
    tag = re.sub(r'(width|height)="([0-9.]+)pt"', r'\1="\2"', tag)
    return f'<svg x="{x_mm * MM_TO_PT:.3f}" y="{y_mm * MM_TO_PT:.3f}" {tag}>{body}'


def assemble(name: str, spec: dict) -> Path:
    width = spec["width_mm"]
    parts, bottom = [], 0.0
    labelled = any(p.get("label") for p in spec["panels"])
    rows = sorted({p["y"] for p in spec["panels"]})
    for i, p in enumerate(spec["panels"]):
        w, h = _panel_size_mm(p["id"])
        band = LABEL_BAND_MM * (rows.index(p["y"]) + 1) if labelled and not spec.get("no_band") else 0.0
        x, y = p["x"], p["y"] + band
        bottom = max(bottom, y + h)
        if x + w > width + 0.5:
            raise ValueError(f"{name}: panel {p['id']} ends at {x + w:.1f} mm > width {width} mm")
        parts.append(_nested(PANELS / p["id"] / f"{p['id']}.svg", x, y, f"p{i}_"))
        if p.get("label"):
            lx, ly = x + p.get("label_dx", 0.0), y - LABEL_BAND_MM + p.get("label_dy", 0.3)
            parts.append(
                f'<text x="{lx * MM_TO_PT:.2f}" y="{(ly * MM_TO_PT) + LABEL_PT * 0.8:.2f}" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="{LABEL_PT}" '
                f'font-weight="bold" fill="#000000">{p["label"]}</text>')
    height = spec.get("height_mm", bottom)
    w_pt, h_pt = width * MM_TO_PT, height * MM_TO_PT
    doc = ('<?xml version="1.0" encoding="utf-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{w_pt:.3f}pt" height="{h_pt:.3f}pt" viewBox="0 0 {w_pt:.3f} {h_pt:.3f}">\n'
           '<rect width="100%" height="100%" fill="#FFFFFF"/>\n' + "\n".join(parts) + "\n</svg>\n")
    ASSEMBLED.mkdir(parents=True, exist_ok=True)
    stem = ASSEMBLED / name.lower().replace(" ", "_")
    stem.with_suffix(".svg").write_text(doc)
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
    import cairosvg

    cairosvg.svg2pdf(bytestring=doc.encode(), write_to=str(stem.with_suffix(".pdf")))
    cairosvg.svg2png(bytestring=doc.encode(), write_to=str(stem.with_suffix(".png")), dpi=300)
    print(f"{name}: {width:.0f} x {height:.1f} mm -> {stem}.pdf")
    return stem


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Assemble figure panels into figures.")
    parser.add_argument("names", nargs="*", help="figure names from layouts.json (default: all)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="skip figures whose panels are not built, instead of failing")
    args = parser.parse_args(argv[1:])
    layouts = json.loads((HERE / "layouts.json").read_text())
    names = args.names or [k for k in layouts if not k.startswith("_")]
    unknown = [n for n in names if n not in layouts]
    if unknown:
        print(f"error: unknown figure name(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    missing = {name: [p["id"] for p in layouts[name]["panels"]
                      if not (PANELS / p["id"] / f"{p['id']}.svg").exists()] for name in names}
    missing = {name: ids for name, ids in missing.items() if ids}
    if missing and not args.allow_missing:
        n = sum(len(ids) for ids in missing.values())
        print(f"error: {n} panel(s) not built, in {len(missing)} figure(s); nothing assembled.",
              file=sys.stderr)
        for name, ids in missing.items():
            print(f"  {name}: {', '.join(ids)}", file=sys.stderr)
        print(f"Build the panels (scripts/figures/panels/) into {PANELS}, or pass --allow-missing "
              "to skip these figures.", file=sys.stderr)
        return 1
    for name in names:
        if name in missing:
            print(f"{name}: skipped, panels not built yet: {', '.join(missing[name])}")
            continue
        assemble(name, layouts[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
