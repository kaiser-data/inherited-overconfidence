"""Correction on the second question set (PROTOCOL.md, 2026-11-16 04:1x; post hoc, no new model output).

  python3 loopgain/nqfix.py      # reads results/runs_ext_<fam>.json, writes results/nqfix_stats.json
Students whose level was set on trivia are still overconfident on Natural Questions. Here n of a
student's answers on Natural Questions are treated as checked, a correction is fitted on them and
applied to its other answers: constant shift, log-odds shift, isotonic fit. Readouts on the other
answers: gap, ECE, AUROC, accuracy on the more confident half, and the rule "state at least 70%".
Beside it the wrong-task baseline: the isotonic correction fitted on the same adapter's trivia replies.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from ext import kind, sel50  # noqa: E402
from thresh import in_format, rule  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
FAMS = ("qwen35", "granite41")
SIZES = (50, 100, 250, 500)
DRAWS = 500
T = 70  # the threshold at which the fitted students pass anything at all (Table 23)
CELLS = ("tcal", "scal", "shuf", "own", "tcal_w", "scal_w")
FIXES = ("shift", "logodds", "isotonic")


def acc(rows: list[dict]) -> float:
    return 100 * sum(bool(r["correct"]) for r in rows) / len(rows)


def conf(rows: list[dict]) -> float:
    return sum(r["conf"] for r in rows) / len(rows)


def _logit(c: float) -> float:
    p = min(99.0, max(1.0, c)) / 100
    return math.log(p / (1 - p))


def logodds_offset(cal: list[dict]) -> float:
    """The one offset b for which the mean of sigmoid(logit(conf) + b) on the checked answers equals
    their accuracy (the maximum-likelihood offset at slope 1). Bounded at +-12 when all are right or wrong."""
    z, target = [_logit(r["conf"]) for r in cal], acc(cal) / 100
    lo, hi = -12.0, 12.0
    for _ in range(60):
        b = (lo + hi) / 2
        if sum(1 / (1 + math.exp(-(x + b))) for x in z) / len(z) < target:
            lo = b
        else:
            hi = b
    return (lo + hi) / 2


def fit(kind_: str, cal: list[dict]):
    """A function conf -> corrected conf (0 to 100), fitted on the checked answers only."""
    if kind_ == "shift":
        d = acc(cal) - conf(cal)
        return lambda c: min(100.0, max(0.0, c + d))
    if kind_ == "logodds":
        b = logodds_offset(cal)
        return lambda c: 100 / (1 + math.exp(-(_logit(c) + b)))
    blocks = core.isotonic_fit([r["conf"] for r in cal], [1.0 if r["correct"] else 0.0 for r in cal])
    return lambda c: 100 * core.isotonic_predict(blocks, c)


def split(rows: list[dict], n: int, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """n checked answers and the rest; no answer is in both."""
    order = rows[:]
    rng.shuffle(order)
    return order[:n], order[n:]


def readout(rows: list[dict]) -> dict:
    """rows: replies in format."""
    c, y = [r["conf"] for r in rows], [bool(r["correct"]) for r in rows]
    t = rule(rows, T)
    return {"gap": conf(rows) - acc(rows), "ece": core.ece(c, y), "auroc": core.auroc(c, y), "sel50": sel50(rows),
            "passed": t["passed"], "wrong_share": t["wrong_share"], "levels": len(set(c))}


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def corrected_draws(rows: list[dict], n: int, draws: int = DRAWS) -> dict:
    rng, got = random.Random(n), {k: [] for k in FIXES}
    for _ in range(draws):
        cal, rest = split(rows, n, rng)
        for k in FIXES:
            f = fit(k, cal)
            got[k].append(readout([dict(r, conf=f(r["conf"])) for r in rest]))
    out = {}
    for k, v in got.items():
        g = sorted(x["gap"] for x in v)
        out[k] = {"within5": 100 * sum(abs(x) <= 5 for x in g) / len(g), "gap_lo": g[int(0.025 * len(g))],
                  "gap_hi": g[int(0.975 * len(g))], "median_abs": sorted(abs(x) for x in g)[len(g) // 2],
                  **{m: _mean([x[m] for x in v]) for m in ("ece", "auroc", "sel50", "passed", "wrong_share", "levels")},
                  "none_passed": 100 * sum(x["wrong_share"] is None for x in v) / len(v)}
    return out


def cell(run: dict) -> dict:
    out = {}
    for rd, key, sizes in (("sampled", "eval2_sampled", SIZES), ("greedy", "eval2", (250,))):
        allr = run["eval"][key]
        rows = [r for r in allr if in_format(r)]
        triv = [r for r in run["eval"]["eval_sampled" if rd == "sampled" else "eval"] if in_format(r)]
        f = fit("isotonic", triv)
        out[rd] = {"n": len(rows), "in_format": 100 * len(rows) / len(allr), "accuracy": acc(rows), "conf": conf(rows),
                   "before": readout(rows),
                   "trivia_fitted": {**readout([dict(r, conf=f(r["conf"])) for r in rows]), "n_fit": len(triv)},
                   "after": {str(n): corrected_draws(rows, n) for n in sizes}}
    return out


def expectations(c: dict) -> dict:
    a = {k: v["sampled"]["after"]["250"] for k, v in c.items()}
    b = {k: v["sampled"]["before"] for k, v in c.items()}
    return {
        "N1 n=250: within 5 points in >= 80% of draws, every cell and correction": all(
            a[k][x]["within5"] >= 80 for k in a for x in FIXES),
        "N2 n=250: AUROC within 0.01 (shifts) and 0.03 (isotonic) of uncorrected": all(
            abs(a[k][x]["auroc"] - b[k]["auroc"]) <= (0.03 if x == "isotonic" else 0.01) for k in a for x in FIXES),
        "N3 n=250: mean ECE after <= 8, every cell and correction": all(a[k][x]["ece"] <= 8 for k in a for x in FIXES),
        "N4 trivia-fitted correction leaves |gap| > 10, every cell": all(
            abs(v["sampled"]["trivia_fitted"]["gap"]) > 10 for v in c.values()),
    }


def main():
    out = {}
    nn = lambda v, f="5.1f": format(float("nan") if v is None else v, f)
    for fam in FAMS:
        f = RES / f"runs_ext_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        c = {kind(name): cell(r) for name, r in runs.items()}
        c = {k: c[k] for k in CELLS if k in c}
        out[fam] = {"cells": c, "expectations": expectations(c)}
        for rd in ("sampled", "greedy"):
            print(f"\n{fam} {rd}\n{'cell':7s} {'fix':9s} {'n':>4s} | within5  gap 95%        ECE  AUROC  half | >={T}: pass wrong")
            for k, e in c.items():
                e = e[rd]
                for lab, r in (("before", e["before"]), ("trivia", e["trivia_fitted"])):
                    print(f"{k:7s} {lab:9s} {'':>4s} | {'':7s} {r['gap']:+6.1f} {'':8s} {r['ece']:5.1f} {r['auroc']:.3f} {r['sel50']:5.1f} | "
                          f"{r['passed']:5.1f} {nn(r['wrong_share'])}   in format {e['in_format']:.1f}% acc {e['accuracy']:.1f}")
                for n, by in e["after"].items():
                    for x in FIXES:
                        r = by[x]
                        print(f"{k:7s} {x:9s} {n:>4s} | {r['within5']:7.1f} ({r['gap_lo']:+5.1f},{r['gap_hi']:+5.1f}) {r['ece']:5.1f} "
                              f"{r['auroc']:.3f} {r['sel50']:5.1f} | {r['passed']:5.1f} {nn(r['wrong_share'])}")
        for k, v in out[fam]["expectations"].items():
            print(f"{fam:10s} {'as expected' if v else 'NOT as expected'}  {k}")
    (RES / "nqfix_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
