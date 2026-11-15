"""Second-provider check: the hosted panel's first turn, served by Adaption instead of Nebius.

Exploratory, added 15 Nov 2026 16:30 CET, after the protocol's stopping time and after the
Nebius panel was known. Same system prompt, questions (E), parser, scorer, temperature 0 and
800-token limit as panel.py. No pushback turn (the gap uses the first reply only).

Only the models that both providers list are run. Written down before the first full run:
we call a gap "reproduced" if it differs from the Nebius gap by less than 5 points on the
questions where both providers' replies were in format.

Needs the `adaption` package (>= 0.9) and ADAPTION_API_KEY (environment or .env):

  python3 panel_adaption.py --n 20        # smoke test
  python3 panel_adaption.py               # resumable: finished (model, id) pairs are skipped
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

RES = HERE.parent / "results"
OUT = RES / "panel_adaption"
MAX_TOKENS = 800
# Adaption model name -> Nebius model id in results/panel/
PAIRS = {"qwen3.8-27b": "Qwen/Qwen3.8-27B",
         "glm-5.2": "zai-org/GLM-5.2",
         "glm-53-flash": "zai-org/GLM-5.3-Flash"}


def client():
    from adaption import Adaption
    if "ADAPTION_API_KEY" not in os.environ:
        for line in (HERE.parent / ".env").read_text().splitlines():
            if line.startswith("ADAPTION_API_KEY="):
                os.environ["ADAPTION_API_KEY"] = line.split("=", 1)[1].strip()
    return Adaption(max_retries=4, timeout=180)


def one(c, model: str, it: dict) -> dict:
    row = {"id": it["id"], "raw": None, "answer": None, "conf": None, "correct": False,
           "usage": {}}
    try:
        r = c.responses.create(model=model, input=core.messages(it["q"]), temperature=0,
                               max_output_tokens=MAX_TOKENS, store=False)
        row["raw"] = r.output_text or ""
        u = r.usage
        det = (getattr(u, "output_tokens_details", None) or {}) if u else {}
        if not isinstance(det, dict):
            det = det.__dict__
        row["usage"] = {"input": u.input_tokens, "output": u.output_tokens,
                        "reasoning": det.get("reasoning_tokens"),
                        "cost": getattr(u, "cost", None)} if u else {}
    except Exception as e:  # recorded; the row is retried on the next run
        row["usage"] = {"error": repr(e)[:300]}
        return row
    p = core.parse(row["raw"])
    if p:
        row["answer"], row["conf"] = p
        row["correct"] = core.is_correct(p[0], it["aliases"])
    return row


def run_model(c, model: str, items: list[dict], workers: int) -> list[dict]:
    f = OUT / (model + ".jsonl")
    done = {}
    if f.exists():
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("raw") is not None:
                done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(lambda it: one(c, model, it), todo):
                done[r["id"]] = r
        with open(f, "w") as fh:
            for it in items:
                if it["id"] in done:
                    fh.write(json.dumps(done[it["id"]], ensure_ascii=False) + "\n")
    return [done[it["id"]] for it in items if it["id"] in done]


def short(m: dict) -> dict:
    return {k: m.get(k) for k in ("n", "n_parsed", "parsed", "accuracy", "mean_conf", "gap",
                                  "auroc")}


def compare(a_rows: list[dict], nebius_id: str, ids: list[str]) -> dict:
    """Both providers on the same questions, and on those where both replies were in format."""
    nb = {}
    for line in (RES / "panel" / (nebius_id.replace("/", "__") + ".jsonl")).read_text().splitlines():
        r = json.loads(line)
        nb[r["id"]] = r
    ad = {r["id"]: r for r in a_rows}
    ids = [i for i in ids if i in ad and i in nb]
    ok = lambda r: r.get("answer") is not None and r.get("conf") is not None
    both = [i for i in ids if ok(ad[i]) and ok(nb[i])]
    out = {"adaption": short(core.metrics([ad[i] for i in ids])),
           "nebius": short(core.metrics([nb[i] for i in ids])),
           "n_both_parsed": len(both)}
    if both:
        ma, mn = core.metrics([ad[i] for i in both]), core.metrics([nb[i] for i in both])
        out["both_parsed"] = {
            "adaption": short(ma), "nebius": short(mn),
            "gap_difference": ma["gap"] - mn["gap"],
            "same_answer": 100 * sum(core.same_answer(ad[i]["answer"], nb[i]["answer"])
                                     for i in both) / len(both),
            "same_correct": 100 * sum(ad[i]["correct"] == nb[i]["correct"] for i in both) / len(both),
            "mean_abs_conf_difference": sum(abs(ad[i]["conf"] - nb[i]["conf"]) for i in both) / len(both),
        }
    out["cost_reported"] = sum((r["usage"].get("cost") or 0) for r in a_rows)
    out["errors"] = sum("error" in r["usage"] for r in a_rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PAIRS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = client()
    items = json.loads((RES / "runs.json").read_text())["data"]["E"][: a.n]
    ids = [it["id"] for it in items]
    summary = {}
    for model in a.models.split(","):
        t0 = time.time()
        rows = run_model(c, model, items, a.workers)
        s = compare(rows, PAIRS[model], ids)
        summary[model] = s
        print(f"{model:14s} adaption acc {s['adaption'].get('accuracy', float('nan')):.1f} "
              f"conf {s['adaption'].get('mean_conf', float('nan')):.1f} "
              f"gap {s['adaption'].get('gap', float('nan')):+.1f} parsed {s['adaption']['parsed']:.2f} | "
              f"nebius gap {s['nebius'].get('gap', float('nan')):+.1f} parsed {s['nebius']['parsed']:.2f} | "
              f"errors {s['errors']} cost {s['cost_reported']:.4f} {time.time() - t0:.0f}s", flush=True)
    if a.n == 1000:
        (RES / "panel_adaption.json").write_text(json.dumps(summary, indent=1))
    else:
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
