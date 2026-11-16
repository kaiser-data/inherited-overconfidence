"""Held-out correction of the trained multiple-choice students (PROTOCOL.md, 2026-11-16 03:15; post hoc).

  python3 loopgain/mcrecal.py     # reads results/runs_mc_<fam>.json, writes results/mc_recal.json
The adapters were not kept, so the 1,000 evaluation questions are split: the isotonic correction of a
trained student's own correctness on its stated confidence is fitted on 500 questions and applied to
the other 500, and the reverse. No question is corrected by a fit that saw it. Sampled answer.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from mcx import FAMS, RES, kind  # noqa: E402

CELLS = ("raw", "tcal", "scal", "shuf", "tcal_s2", "scal_s2")
SPLITS = range(2, 202)


def ok(r: dict) -> bool:
    return r.get("answer") is not None and r.get("conf") is not None


def cross_fit(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Corrected rows of the two halves; each half is corrected by the fit on the other."""
    by = {r["id"]: r for r in rows}
    perm = core.seeded_perm(sorted(by), seed)
    halves = [[by[i] for i in perm[:len(perm) // 2]], [by[i] for i in perm[len(perm) // 2:]]]
    out = []
    for fit, apply in ((halves[1], halves[0]), (halves[0], halves[1])):
        cal = [r for r in fit if ok(r)]
        blocks = core.isotonic_fit([r["conf"] for r in cal], [1.0 if r["correct"] else 0.0 for r in cal])
        out.append([dict(r, conf=100 * core.isotonic_predict(blocks, r["conf"])) for r in apply if ok(r)])
    return out[0], out[1]


def gap(rows: list[dict]) -> float:
    return sum(r["conf"] for r in rows) / len(rows) - 100 * sum(bool(r["correct"]) for r in rows) / len(rows)


def boot(rows: list[dict], n: int = 2000, seed: int = 1) -> tuple[float, float]:
    rng = random.Random(seed)
    g = sorted(gap([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return g[int(0.025 * n)], g[int(0.975 * n)]


def cell(rows: list[dict]) -> dict:
    before = [r for r in rows if ok(r)]
    a, b = cross_fit(rows, 1)
    after = a + b
    lo, hi = boot(after)
    m0, m1 = core.metrics(before), core.metrics(after)
    many = sorted(gap([r for h in cross_fit(rows, s) for r in h]) for s in SPLITS)
    return {"n": len(after), "gap_before": m0["gap"], "gap_after": m1["gap"], "gap_lo": lo, "gap_hi": hi,
            "gap_half_a": gap(a), "gap_half_b": gap(b), "ece_before": m0["ece"], "ece_after": m1["ece"],
            "auroc_before": m0["auroc"], "auroc_after": m1["auroc"], "accuracy": m0["accuracy"],
            "conf_before": m0["mean_conf"], "conf_after": m1["mean_conf"], "levels_after": len({round(r["conf"], 6) for r in after}),
            "splits_median": many[len(many) // 2], "splits_lo": many[int(0.025 * len(many))], "splits_hi": many[int(0.975 * len(many))]}


def main():
    out = {}
    for fam in FAMS:
        f = RES / f"runs_mc_{fam}.json"
        if not f.exists():
            continue
        runs = {kind(n): r for n, r in json.loads(f.read_text())["runs"].items()}
        c = {k: cell(runs[k]["eval"]["eval_sampled"]) for k in CELLS if k in runs}
        exp = {"X1 corrected |gap| <= 5 in every cell": all(abs(e["gap_after"]) <= 5 for e in c.values()),
               "X2 each half |gap| <= 8": all(max(abs(e["gap_half_a"]), abs(e["gap_half_b"])) <= 8 for e in c.values()),
               "X3 AUROC after within 0.03 of before": all(abs(e["auroc_after"] - e["auroc_before"]) <= 0.03 for e in c.values())}
        out[fam] = {"cells": c, "expectations": exp}
        print(f"\n{fam}\n{'cell':8s} before  after (95%)          half a half b | ece b/a      | auc b/a     | lv | 200 splits")
        for k, e in c.items():
            print(f"{k:8s} {e['gap_before']:+6.1f} {e['gap_after']:+6.1f} ({e['gap_lo']:+5.1f},{e['gap_hi']:+5.1f}) {e['gap_half_a']:+6.1f} {e['gap_half_b']:+6.1f} | "
                  f"{e['ece_before']:5.1f} {e['ece_after']:5.1f} | {e['auroc_before']:.2f} {e['auroc_after']:.2f} | {e['levels_after']:2d} | "
                  f"{e['splits_median']:+5.1f} ({e['splits_lo']:+5.1f},{e['splits_hi']:+5.1f})")
        for k, v in exp.items():
            print(f"{fam:10s} {'as expected' if v else 'NOT as expected'}  {k}")
    (RES / "mc_recal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
