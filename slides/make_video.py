#!/usr/bin/env python3
"""Render a narrated video from the slide deck.

Usage: python3 slides/make_video.py [voice]
Frames: headless Chrome screenshots of slides/confidence-slider-deck.html (1920x1080).
Narration: the "Say" notes of each slide, spoken by the macOS `say` voice (synthetic).
Slide 4 is shown in its three slider settings. Writes slides/confidence-slider-video.mp4.
"""
import html
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECK = ROOT / "slides/confidence-slider-deck.html"
SRC = ROOT / "slides/src/deck.src.html"
OUT = ROOT / "slides/confidence-slider-video.mp4"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VOICE = sys.argv[1] if len(sys.argv) > 1 else "Daniel"
PAUSE = 0.7  # seconds of silence after each slide


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout
    return float(out)


def shot(url, png):
    run(CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
        "--virtual-time-budget=3000", f"--screenshot={png}", url)


notes = [html.unescape(re.sub(r"<[^>]+>", "", n)).strip()
         for n in re.findall(r'<aside class="notes"><b>Say</b>(.*?)</aside>', SRC.read_text(), re.S)]
assert len(notes) == 9, len(notes)

tmp = pathlib.Path(tempfile.mkdtemp(prefix="deckvideo-"))
segments = []  # (png, seconds)
audio = []
for i, text in enumerate(notes, 1):
    aiff = tmp / f"a{i}.aiff"
    run("say", "-v", VOICE, "-r", "165", "-o", str(aiff), text)
    wav = tmp / f"a{i}.wav"
    run("ffmpeg", "-y", "-i", str(aiff), "-af", f"apad=pad_dur={PAUSE}", "-ar", "44100", "-ac", "2", str(wav))
    secs = duration(wav)
    audio.append(wav)
    if i == 4:  # the slider: teacher's own -> calibrated -> fitted to the student
        for k, dial in enumerate((2, 1, 0)):
            png = tmp / f"s4-{k}.png"
            shot(f"file://{DECK}?dial={dial}#s4", png)
            segments.append((png, secs * (0.30, 0.25, 0.45)[k]))
    else:
        png = tmp / f"s{i}.png"
        shot(f"file://{DECK}#s{i}", png)
        segments.append((png, secs))

frames = tmp / "frames.txt"
frames.write_text("".join(f"file '{p}'\nduration {d:.3f}\n" for p, d in segments) + f"file '{segments[-1][0]}'\n")
sound = tmp / "sound.txt"
sound.write_text("".join(f"file '{w}'\n" for w in audio))
run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(frames), "-f", "concat", "-safe", "0", "-i", str(sound),
    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-b:a", "128k",
    "-shortest", "-movflags", "+faststart", str(OUT))
print(OUT, f"{duration(OUT):.0f} s", f"{OUT.stat().st_size / 1e6:.1f} MB", "voice:", VOICE)
