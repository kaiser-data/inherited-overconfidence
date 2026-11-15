#!/usr/bin/env bash
# numbers -> figures -> filled LaTeX -> official Apart template (.docx) -> PDF. Prints page and word checks.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; W="$D/build"; mkdir -p "$W"
TITLE="Inherited Overconfidence: Whether Audited Data Corrects It Depends on Decoding and on Tagging the Audits"
[ -f "$W/template.docx" ] || curl -sL -o "$W/template.docx" "https://docs.google.com/document/d/17b3JWki-ZkaqdmqGI5d0gof-HwlGuWU9Mc1_klNZXKs/export?format=docx"
if [ "${SKIP_ANALYSIS:-0}" != "1" ]; then  # SKIP_ANALYSIS=1: text-only rebuild from the existing summaries
python3 "$D/../loopgain/analyze.py" > "$W/analyze.log"
[ -f "$D/../results/runs_llama31.json" ] && python3 "$D/../loopgain/analyze.py" llama31 > "$W/analyze_llama31.log" || true
fi
python3 "$D/make_numbers.py" > "$W/numbers.log"
python3 "$D/figures.py" > "$D/figure-numbers.txt"
python3 "$D/fill.py"
python3 "$D/tools/prep_docx_tex.py" "$W/main.filled.tex" "$W/docx.tex"
python3 - "$W" <<'PY'
import re, subprocess, sys
w = sys.argv[1]; t = open(f"{w}/docx.tex").read()
pre = re.sub(r"^\\(title|author|date)\{.*$", "", t.split("\\begin{document}")[0], flags=re.M)
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1).replace("\\noindent", "")
body = t.split("\\end{abstract}", 1)[1]
open(f"{w}/body.tex", "w").write(pre + "\\begin{document}\n" + body)
txt = subprocess.run(["pandoc", "-f", "latex", "-t", "plain", "--wrap=none"], input=ab, capture_output=True, text=True).stdout
open(f"{w}/abstract.txt", "w").write(txt.strip())
print("abstract words:", len(txt.split()))
PY
python3 "$D/tools/build_template.py" "$W/template.docx" "$W/body.tex" "$D/paper.docx" \
  --title "$TITLE" --author "Martin Kaiser" --affil "Independent researcher" --abstract "$W/abstract.txt" --resource-path "$D"
soffice --headless --convert-to pdf --outdir "$D" "$D/paper.docx" >/dev/null 2>&1
echo "pages: $(pdfinfo "$D/paper.pdf" | awk '/Pages/{print $2}')"
pdftotext "$D/paper.pdf" - | awk 'BEGIN{p=1} /\f/{p++} /^[0-9]\. [A-Z]|^References$|^Appendix [A-C]|^Code and Data|^LLM Usage/{print p": "$0}'
