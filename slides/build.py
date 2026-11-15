#!/usr/bin/env python3
"""Build the self-contained slide deck: inline fonts and figures as data URIs.

Usage: python3 slides/build.py
Reads slides/src/deck.src.html, writes slides/confidence-slider-deck.html.
{{FONT:path}} resolves against the martin-uix skill, {{IMG:path}} against the repo root.
"""
import base64
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKILL = pathlib.Path.home() / ".claude/skills/martin-uix"
SRC = ROOT / "slides/src/deck.src.html"
OUT = ROOT / "slides/confidence-slider-deck.html"


def uri(path: pathlib.Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


html = SRC.read_text()
html = re.sub(r"\{\{FONT:([^}]+)\}\}", lambda m: uri(SKILL / m.group(1), "font/woff2"), html)
html = re.sub(r"\{\{IMG:([^}]+)\}\}", lambda m: uri(ROOT / m.group(1), "image/png"), html)
assert "{{" not in html, "placeholder left"
OUT.write_text(html)
print(OUT, f"{len(html) / 1024:.0f} KB")
