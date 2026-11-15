"""Stage narration into edit/public and write edit/src/manifest.json.

  python3 prep.py --voice m-calm     # reads out/tts/<voice>/<line-id>.wav (picked takes)
One shared gain for every line, gentle compression; silence trimmed at both ends; 48 kHz.
"""
import argparse
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
PUB = ROOT / "edit" / "public"
FONTS = pathlib.Path.home() / ".claude/skills/martin-uix/assets/fonts"


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode().strip())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--voice", default="m-calm"); a = ap.parse_args()
    script = json.loads((ROOT / "script.json").read_text())
    (PUB / "vo").mkdir(parents=True, exist_ok=True); (PUB / "fonts").mkdir(parents=True, exist_ok=True)
    for f in ("plex-serif-600", "plex-sans-400", "plex-sans-600", "plex-mono-500"):
        shutil.copy2(FONTS / f"{f}.woff2", PUB / "fonts" / f"{f}.woff2")
    lines = {}
    for ln in script["lines"]:
        src = ROOT / "out" / "tts" / a.voice / f"{ln['id']}.wav"
        dst = PUB / "vo" / f"{ln['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af",
                        "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                        "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
                        "acompressor=threshold=-24dB:ratio=2:attack=25:release=300,volume=6dB,alimiter=limit=0.89",
                        "-ar", "48000", str(dst)], check=True)
        lines[ln["id"]] = {"file": f"vo/{dst.name}", "dur": round(dur(dst), 3), "text": ln["text"]}
    manifest = {"voice": a.voice, "lines": lines,
                "disclosure": "Narration: AI-generated voice (Qwen3-TTS). Charts are drawn from the paper's data; no footage is generated."}
    (ROOT / "edit" / "src" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print({i: x["dur"] for i, x in lines.items()}, "total", round(sum(x["dur"] for x in lines.values()), 1))


main()
