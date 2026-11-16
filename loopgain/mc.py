"""Multiple-choice arm (PROTOCOL.md, 2026-11-16): the lead experiment on MMLU-Pro questions.

  python3 mc.py gen --set MP            # hosted teacher on the 200 pilot questions (resumable)
  python3 mc.py gen --set MD            # ... on the 2,000 record questions
  python3 mc.py build                   # label sets -> results/dist/mc_<name>/d_main.jsonl

The partition is our own cut of the MMLU-Pro test split (modal_app.prep_mc); results are not
benchmark scores. Label logic is the distillation arm's (distill.tcal / scal / shuf).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import core  # noqa: E402
import distill  # noqa: E402

RES = HERE.parent / "results"
MCD = RES / "mc"
TEACHER = config.DIST_SEED_TEACHER
distill.MAX_TOKENS = 8000  # the teacher reasons longer on these questions (pilot: 6% ended at 4,000)


def _rows(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def to_row(item: dict, raw: str | None) -> dict:
    """The teacher's reply; the answer is reduced to the option letter (None if it names none)."""
    p = core.parse(distill.clean(raw) or "")
    ans, conf = p if p else (None, None)
    let = core.letter(ans)
    return {"id": item["id"], "raw": raw, "answer": let, "conf": conf if let else None,
            "correct": bool(let is not None and let == item["value"])}


def gen(qset: str, workers: int):
    import panel
    k, model = panel.key(), distill.HOSTED[TEACHER]
    items = _rows(MCD / f"{qset}.jsonl")
    f = MCD / "_hosted" / f"{TEACHER}_{qset}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in _rows(f) if r.get("raw") is not None} if f.exists() else {}

    def one(it):
        r = distill._chat(model, core.messages(it["q"], core.SYSTEM_MC), k)
        return {**to_row(it, r.get("raw")), **{kk: v for kk, v in r.items() if kk != "raw"},
                "t": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, [it for it in items if it["id"] not in done]):
            done[r["id"]] = r
    rows = [done[it["id"]] for it in items if it["id"] in done]
    with open(f, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r["answer"] is not None]
    fin: dict = {}
    for r in rows:
        fin[str(r.get("finish"))] = fin.get(str(r.get("finish")), 0) + 1
    print(json.dumps({"set": qset, "model": model, "asked": len(items), "answered": len(rows),
                      "parsed": len(ok) / max(1, len(items)),
                      "accuracy_parsed": 100 * sum(r["correct"] for r in ok) / max(1, len(ok)),
                      "accuracy_unparsed_wrong": 100 * sum(r["correct"] for r in rows) / max(1, len(items)),
                      "mean_conf": sum(r["conf"] for r in ok) / max(1, len(ok)), "finish": fin,
                      "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)}, indent=1))


def build():
    """RAW, TCAL, SCAL (per student) and SHUF for the record questions MD."""
    D = _rows(MCD / "MD.jsonl")
    q = {r["id"]: r for r in D}
    perm = core.seeded_perm([r["id"] for r in D], config.DATA_SEED)
    allrows = _rows(MCD / "_hosted" / f"{TEACHER}_MD.jsonl")
    rows = [r for r in allrows if r.get("answer") is not None and r.get("conf") is not None]
    sets = {"raw": {r["id"]: r["conf"] for r in rows}, "tcal": distill.tcal(rows, q, perm)}
    sc = {}
    for fam in config.NEW_FAMS:
        src = MCD / "_base" / f"{fam}_d_main.jsonl"
        if src.exists():
            sc[fam] = {r["id"]: bool(r["correct"]) for r in _rows(src)}
            sets[f"scal_{fam}"] = distill.scal(rows, sc[fam])
    sets["shuf"] = distill.shuf(sets["tcal"])
    info = {"n_in": len(allrows), "retained": len(rows) / len(D), "sets": {}}
    for name, labels in sets.items():
        out = distill.relabel(rows, labels)
        fam = name[5:] if name.startswith("scal_") else None
        st = distill.stats(out, sc[fam] if fam else None)
        assert st["answers_sha"] == distill.answers_hash(rows), name
        d = distill.OUT / f"mc_{TEACHER}__{name}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "d_main.jsonl", "w") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        info["sets"][name] = st
        print(f"mc_{TEACHER}__{name:16s} n {st['n']} label mean {st['label_mean']:5.1f} mode {st['label_mode']} "
              f"({st['mode_share']:.2f}) teacher acc {st['teacher_accuracy']:5.1f}"
              + (f" student-base acc {st['student_base_accuracy']:5.1f}" if fam else ""))
    (MCD / "labels.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "build"])
    ap.add_argument("--set", default="MP")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    gen(a.set, a.workers) if a.cmd == "gen" else build()
