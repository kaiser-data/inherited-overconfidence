"""Substitute the macros of numbers.tex and text_macros.tex into main.tex -> build/main.filled.tex."""
import re
from pathlib import Path

D = Path(__file__).parent
macros = {}
for f in ("numbers.tex", "text_macros.tex"):
    src = (D / f).read_text()
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{", src):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        macros[m.group(1)] = src[m.end():i - 1]
tex = (D / "main.tex").read_text()
pat = re.compile(r"\\([A-Za-z]+)(\{\})?")
for _ in range(4):  # text macros contain number macros
    tex = pat.sub(lambda m: macros[m.group(1)] if m.group(1) in macros else m.group(0), tex)
bad = sorted(set(re.findall(r"n/a|pending|to be added", tex)))
(D / "build" / "main.filled.tex").write_text(tex)
print(f"filled {len(macros)} macros; open items: {bad or 'none'}")
