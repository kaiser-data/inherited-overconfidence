"""Narration on Modal: Qwen3-TTS (Apache-2.0). Per voice: design one reference take from a text persona
(VoiceDesign model), then clone it for every script line (Base model), TAKES seeds each.

  modal run modal/voice.py
Outputs: out/tts/<voice>-takes/<line-id>-t<k>.wav, out/tts/<voice>-ref.wav ; pick with modal/pick_qwen.py.
"""
import json
import pathlib
import time

import modal

app = modal.App("slider-video-voice")
hf_cache = modal.Volume.from_name("astrolab-hf-cache", create_if_missing=True)
ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "sox")
    .pip_install("qwen-tts", "soundfile", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
VOICES = {
    "m-calm": "A mature male science narrator in his late forties. Warm baritone, calm authority, curious and friendly, "
              "an unhurried documentary pace, clear neutral English accent. Never shouting.",
    "f-calm": "A calm, warm female science presenter in her forties. Grounded, clear and curious, slightly lower voice, "
              "an unhurried natural pace, clear neutral English accent. Never shouting.",
}
REF_TEXT = ("Knowing something includes knowing how sure you can be. In this short film we look at where the "
            "confidence of a small AI model comes from, and what happens when it is changed.")
TAKES = 3


@app.function(image=image, gpu="L40S", volumes={"/cache": hf_cache}, timeout=2400)
def speak(voice: str, instruct: str, lines: list[dict]) -> dict:
    import io
    import tempfile

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    def wav(arr, sr):
        buf = io.BytesIO(); sf.write(buf, arr, sr, format="WAV"); return buf.getvalue()

    d = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")
    torch.manual_seed(1)
    ref, sr = d.generate_voice_design(text=REF_TEXT, language="English", instruct=instruct)
    out = {"ref": wav(ref[0], sr)}
    del d
    torch.cuda.empty_cache()
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        fh.write(out["ref"]); fh.flush()
        prompt = m.create_voice_clone_prompt(ref_audio=fh.name, ref_text=REF_TEXT, x_vector_only_mode=False)
    for ln in lines:
        for k in range(TAKES):
            torch.manual_seed(100 + k)
            w, sr = m.generate_voice_clone(text=ln["text"], language="English", voice_clone_prompt=prompt)
            out[f"{ln['id']}-t{k}"] = wav(w[0], sr)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(voices: str = "m-calm,f-calm"):
    t0 = time.time()
    lines = json.loads((ROOT / "script.json").read_text())["lines"]
    handles = {v: speak.spawn(v, VOICES[v], lines) for v in voices.split(",")}
    for v, h in handles.items():
        res = h.get()
        dest = ROOT / "out" / "tts" / f"{v}-takes"; dest.mkdir(parents=True, exist_ok=True)
        (ROOT / "out" / "tts" / f"{v}-ref.wav").write_bytes(res.pop("ref"))
        for k, b in res.items():
            (dest / f"{k}.wav").write_bytes(b)
        with (ROOT / "out" / "runs.jsonl").open("a") as fh:
            fh.write(json.dumps({"job": "tts", "model": "qwen3-tts-1.7b voicedesign+base", "voice": v, "gpu": "L40S",
                                 "lines": len(lines), "takes": TAKES, "seconds": round(time.time() - t0), "ts": time.time()}) + "\n")
        print(v, len(res), "files", round(time.time() - t0), "s")
