"""Threshold reading of Result 2 (PROTOCOL.md, 2026-11-16 01:38; post hoc, no new model output).

  python3 loopgain/thresh.py     # reads results/runs_dist_<fam>.json, writes results/thresh_stats.json
What one fixed rule, "act on an answer only if the model states at least T%", does with students that
differ only in the confidence labels of their records. Lead experiment, DeepSeek-V4.1-Flash records,
seed 1, sampled answer, the 1,000 test questions E. Per cell and threshold: share of the questions
passed, share of the passed answers that are wrong, wrong answers passed per 1,000 questions, share of
the right answers held back; and the wrong share on the more confident half (equal coverage).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import core  # noqa: E402
from ext import sel50  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
FAMS = config.NEW_FAMS
TEACHER = config.DIST_SEED_TEACHER
THRESHOLDS = (70, 80, 90)  # 80 is the rule; 70 and 90 are reported beside it whatever they show
CELLS = ("raw", "tcal", "scal", "shuf", "own", "tcal_recal")


def in_format(r: dict) -> bool:
    return r.get("answer") is not None and r.get("conf") is not None


def rule(rows: list[dict], t: float) -> dict:
    """One reply per question. Passed: in format and stated confidence >= t."""
    passed = [r for r in rows if in_format(r) and r["conf"] >= t]
    wrong = sum(not r["correct"] for r in passed)
    right_all = sum(bool(r["correct"]) for r in rows if in_format(r))
    return {"passed": 100 * len(passed) / len(rows),
            "wrong_share": 100 * wrong / len(passed) if passed else None,
            "wrong_per_1000": 1000 * wrong / len(rows),
            "right_held_back": 100 * (right_all - (len(passed) - wrong)) / right_all if right_all else None}


def interval(rows: list[dict], t: float, n: int = 2000, seed: int = 1) -> dict:
    """Question bootstrap for the share passed and the wrong share among the passed answers."""
    flags = [(in_format(r) and r["conf"] >= t, not r["correct"]) for r in rows]
    rng, m, ps, ws = random.Random(seed), len(rows), [], []
    for _ in range(n):
        p = w = 0
        for _ in range(m):
            f = flags[rng.randrange(m)]
            if f[0]:
                p += 1
                w += f[1]
        ps.append(100 * p / m)
        if p:
            ws.append(100 * w / p)
    ps.sort()
    ws.sort()
    q = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v))]) if v else (None, None)
    return {"passed_lo": q(ps)[0], "passed_hi": q(ps)[1], "wrong_lo": q(ws)[0], "wrong_hi": q(ws)[1]}


def corrected(cal_rows: list[dict], e_rows: list[dict]) -> list[dict]:
    """The held-out correction of distill.recalibrated: isotonic fit of the trained student's own
    correctness on its stated confidence (2,000 held-out questions), applied to its replies on E."""
    cal = [r for r in cal_rows if in_format(r)]
    blocks = core.isotonic_fit([r["conf"] for r in cal], [1.0 if r["correct"] else 0.0 for r in cal])
    return [dict(r, conf=100 * core.isotonic_predict(blocks, r["conf"])) if in_format(r) else r for r in e_rows]


def cell(rows: list[dict]) -> dict:
    ok = [r for r in rows if in_format(r)]
    out = {"n": len(rows), "in_format": 100 * len(ok) / len(rows),
           "accuracy": 100 * sum(bool(r["correct"]) for r in ok) / len(ok),
           "conf": sum(r["conf"] for r in ok) / len(ok), "half_wrong": 100 - sel50(ok)}
    for t in THRESHOLDS:
        out[f"t{t}"] = {**rule(rows, t), **interval(rows, t)}
    return out


def expectations(c: dict) -> dict:
    """The expectation written down before the computation (not a registered prediction)."""
    t80, t70 = (lambda k: c[k]["t80"]), (lambda k: c[k]["t70"])
    return {
        "E1 at 80, RAW and TCAL pass > 90% and 35-42% of the passed answers are wrong": all(
            t80(k)["passed"] > 90 and 35 <= t80(k)["wrong_share"] <= 42 for k in ("raw", "tcal")),
        "E2 at 80, SCAL and the corrected TCAL student pass < 5%": all(t80(k)["passed"] < 5 for k in ("scal", "tcal_recal")),
        "E3 at 70, SCAL passes 50-70% with a wrong share < 30%": (
            50 <= t70("scal")["passed"] <= 70 and t70("scal")["wrong_share"] is not None and t70("scal")["wrong_share"] < 30),
        "E4 more confident half: TCAL and SCAL within 5 points": abs(c["tcal"]["half_wrong"] - c["scal"]["half_wrong"]) <= 5,
    }


def main():
    out = {}
    for fam in FAMS:
        f = RES / f"runs_dist_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        names = {"raw": f"{TEACHER}__raw", "tcal": f"{TEACHER}__tcal", "scal": f"{TEACHER}__scal_{fam}",
                 "shuf": f"{TEACHER}__shuf", "own": "base"}
        ev = {}
        for k, teacher in names.items():
            run = next(r for r in runs.values() if r["meta"]["teacher"] == teacher and r["meta"]["seed"] == 1)
            ev[k] = run["eval"]
        rows = {k: e["eval_sampled"] for k, e in ev.items()}
        rows["tcal_recal"] = corrected(ev["tcal"]["d_main"], ev["tcal"]["eval_sampled"])
        c = {k: cell(rows[k]) for k in CELLS}
        out[fam] = {"cells": c, "expectations": expectations(c)}
        print(f"\n{fam}\n{'cell':11s}  says   acc | " + " | ".join(f">={t}: pass wrong /1000 held" for t in THRESHOLDS) + " | half wrong")
        for k in CELLS:
            e = c[k]
            print(f"{k:11s} {e['conf']:5.1f} {e['accuracy']:5.1f} | " + " | ".join(
                f"{e[f't{t}']['passed']:9.1f} {(e[f't{t}']['wrong_share'] if e[f't{t}']['wrong_share'] is not None else float('nan')):5.1f} "
                f"{e[f't{t}']['wrong_per_1000']:5.0f} {e[f't{t}']['right_held_back']:4.0f}" for t in THRESHOLDS) + f" | {e['half_wrong']:5.1f}")
        for k, v in out[fam]["expectations"].items():
            print(f"{fam:10s} {'as expected' if v else 'NOT as expected'}  {k}")
    (RES / "thresh_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
