"""Hosted-model panel on Nebius Token Factory: where do deployed models start on the map?

Same system prompt, questions (E), parser and scorer as the trained cells. Temperature 0.
Serving details (quantisation, chat template, hidden reasoning) are not under our control;
this is a descriptive panel, not part of the return-map design.

  python3 panel.py --models all            # resumable: finished (model, id) pairs are skipped
  python3 panel.py --models "Qwen/Qwen3-30B-A3B-Instruct-2507,google/gemma-3-27b-it" --n 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

RES = HERE.parent / "results"
OUT = RES / "panel"
URL = "https://api.tokenfactory.nebius.com/v1"
MAX_TOKENS = 800  # room for models that reason before answering
SKIP = ("Embedding",)


def key() -> str:
    if "NEBIUS_API_KEY" in os.environ:
        return os.environ["NEBIUS_API_KEY"]
    for p in (HERE.parent / ".env", HERE.parent.parent / "when-agents-collude" / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("NEBIUS_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("NEBIUS_API_KEY not found")


def _call(path: str, payload: dict | None, k: str, timeout: int = 120):
    req = urllib.request.Request(URL + path, headers={"Authorization": "Bearer " + k,
                                                      "Content-Type": "application/json"},
                                 data=json.dumps(payload).encode() if payload else None)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat(model: str, msgs: list[dict], k: str) -> tuple[str | None, dict]:
    err = None
    for attempt in range(4):
        try:
            r = _call("/chat/completions", {"model": model, "messages": msgs, "temperature": 0,
                                            "max_tokens": MAX_TOKENS}, k)
            return (r["choices"][0]["message"].get("content") or ""), r.get("usage") or {}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:  # timeouts, connection resets
            err = repr(e)
        time.sleep(2 * (attempt + 1))
    return None, {"error": err}


def one(model: str, it: dict, k: str) -> dict:
    msgs = core.messages(it["q"])
    raw, usage = chat(model, msgs, k)
    row = {"id": it["id"], "raw": raw, "answer": None, "conf": None, "correct": False,
           "raw2": None, "answer2": None, "conf2": None, "correct2": False, "usage": usage}
    if raw is None:
        return row
    p = core.parse(raw)
    if p:
        row["answer"], row["conf"] = p
        row["correct"] = core.is_correct(p[0], it["aliases"])
    raw2, _ = chat(model, msgs + [{"role": "assistant", "content": raw},
                                  {"role": "user", "content": core.PUSHBACK}], k)
    row["raw2"] = raw2
    p2 = core.parse(raw2) if raw2 else None
    if p2:
        row["answer2"], row["conf2"] = p2
        row["correct2"] = core.is_correct(p2[0], it["aliases"])
    return row


def run_model(model: str, items: list[dict], k: str, workers: int) -> dict:
    f = OUT / (model.replace("/", "__") + ".jsonl")
    done = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("raw") is not None:  # failed calls are retried on the next run
                done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(lambda it: one(model, it, k), todo):
                done[r["id"]] = r
        with open(f, "w") as fh:
            for it in items:
                if it["id"] in done:
                    fh.write(json.dumps(done[it["id"]], ensure_ascii=False) + "\n")
    rows = [done[it["id"]] for it in items if it["id"] in done]
    m = core.metrics(rows)
    m["answered"] = sum(r["raw"] is not None for r in rows) / max(1, len(items))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    k = key()
    items = json.loads((RES / "runs.json").read_text())["data"]["E"][: a.n]
    if a.models == "all":
        models = sorted(m["id"] for m in _call("/models", None, k)["data"]
                        if not any(s in m["id"] for s in SKIP))
    else:
        models = a.models.split(",")
    summary = {}
    if (RES / "panel.json").exists() and a.n == 1000:
        summary = json.loads((RES / "panel.json").read_text())
    for model in models:
        t0 = time.time()
        m = run_model(model, items, k, a.workers)
        summary[model] = m
        print(f"{model:45s} answered {m['answered']:.2f} parsed {m.get('parsed', 0):.2f} "
              f"acc {m.get('accuracy', float('nan')):.1f} conf {m.get('mean_conf', float('nan')):.1f} "
              f"gap {m.get('gap', float('nan')):+.1f}  {time.time() - t0:.0f}s", flush=True)
        if a.n == 1000:
            (RES / "panel.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
