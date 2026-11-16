"""Prose arm (PROTOCOL.md, 2026-11-16 05:4x, slimmed 05:5x): does certainty travel in a sentence with no number?

  python3 prose.py     # reads results/runs_prose_<fam>.json, writes results/prose_stats.json
Conditions: A neutral, C hedged, D wording from the calibrated label, E the same wording shuffled.
First turn: one sentence; second turn (sees only the bare answer): a number. Sampled answer is primary.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import core  # noqa: E402

RES = HERE.parent / "results"
N_BOOT = 2000
FORMS = tuple(core.PROSE)


def scored(rows):
    """Replies whose first turn is one of the four sentences and whose second turn gave a number."""
    return [r for r in rows if r.get("form") is not None and r.get("conf") is not None]


def cell(rows: list[dict]) -> dict:
    fmt = [r for r in rows if r.get("form") is not None]
    ok = scored(rows)
    acc = lambda rs: 100 * sum(bool(r["correct"]) for r in rs) / len(rs) if rs else None
    out = {"n": len(rows), "in_format": 100 * len(fmt) / len(rows), "number_given": 100 * len(ok) / max(1, len(fmt)),
           "accuracy": acc(fmt), "accuracy_loose_all": 100 * sum(bool(r.get("correct_loose")) for r in rows) / len(rows),
           "conf": sum(r["conf"] for r in ok) / len(ok) if ok else None,
           "auroc": core.auroc([r["conf"] for r in ok], [bool(r["correct"]) for r in ok]) if ok else None,
           "levels": len({r["conf"] for r in ok}),
           "forms": {f: 100 * sum(r["form"] == f for r in fmt) / max(1, len(fmt)) for f in FORMS},
           "accuracy_by_form": {f: acc([r for r in fmt if r["form"] == f]) for f in FORMS},
           "conf_by_form": {f: (lambda rs: sum(r["conf"] for r in rs) / len(rs) if rs else None)([r for r in ok if r["form"] == f]) for f in FORMS}}
    a, o = [r for r in fmt if r["form"] == "assertive"], [r for r in fmt if r["form"] != "assertive"]
    out["assertive_minus_other_accuracy"] = acc(a) - acc(o) if a and o else None
    out["gap"] = out["conf"] - acc(ok) if ok else None
    return out


def paired(a_rows, b_rows, stat) -> dict:
    """stat(rows of a) - stat(rows of b) on the questions scored in both; question bootstrap."""
    A, B = {r["id"]: r for r in scored(a_rows)}, {r["id"]: r for r in scored(b_rows)}
    ids = sorted(set(A) & set(B))
    d = lambda sel: stat([A[i] for i in sel]) - stat([B[i] for i in sel])
    rng, bs = random.Random(1), []
    for _ in range(N_BOOT):
        try:
            bs.append(d([ids[rng.randrange(len(ids))] for _ in ids]))
        except TypeError:  # a resample without both classes has no AUROC
            pass
    bs.sort()
    return {"d": d(ids), "lo": bs[int(0.025 * len(bs))], "hi": bs[int(0.975 * len(bs))], "n": len(ids)}


mean_conf = lambda rs: sum(r["conf"] for r in rs) / len(rs)
accuracy = lambda rs: 100 * sum(bool(r["correct"]) for r in rs) / len(rs)
auc = lambda rs: core.auroc([r["conf"] for r in rs], [bool(r["correct"]) for r in rs])


def main():
    out = {}
    for fam in config.NEW_FAMS:
        f = RES / f"runs_prose_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        ev = {n[-1]: r["eval"] for n, r in runs.items() if n[-3:-1] == "__p"[-2:] or n[-2] == "p"}
        res = {}
        print(f"\n===== {fam}")
        for rd, key in (("sampled", "eval_sampled"), ("greedy", "eval")):
            c = {k: cell(e[key]) for k, e in sorted(ev.items())}
            d = {}
            for k in ("C", "D", "E"):
                if k in ev and "A" in ev:
                    d[f"conf:{k}-A"] = paired(ev[k][key], ev["A"][key], mean_conf)
                    d[f"acc:{k}-A"] = paired(ev[k][key], ev["A"][key], accuracy)
            if "D" in ev and "E" in ev:
                d["auroc:D-E"] = paired(ev["D"][key], ev["E"][key], auc)
                d["conf:D-E"] = paired(ev["D"][key], ev["E"][key], mean_conf)
            res[rd] = {"cells": c, "diffs": d}
            print(f"{rd}\ncond  in fmt  number  right   says  AUROC lvls | wording % (neutral/assertive/probably/hedged) | right by wording")
            for k, e in c.items():
                fm, ab = e["forms"], e["accuracy_by_form"]
                print(f"  {k}   {e['in_format']:5.1f}  {e['number_given']:5.1f}  {e['accuracy']:5.1f}  {e['conf']:5.1f}  "
                      f"{(e['auroc'] or float('nan')):.3f} {e['levels']:3d}  | " + " / ".join(f"{fm[x]:4.1f}" for x in FORMS)
                      + " | " + " / ".join("  -- " if ab[x] is None else f"{ab[x]:5.1f}" for x in FORMS))
            for k, v in d.items():
                print(f"  {k:12s} {v['d']:+7.3f} ({v['lo']:+.3f},{v['hi']:+.3f}) n {v['n']}")
        S, D = res["sampled"]["cells"], res["sampled"]["diffs"]
        have = lambda *ks: all(k in S for k in ks)
        P = {}
        P["S1 at least 95% of first replies in one of the four sentences"] = all(e["in_format"] >= 95 for e in S.values())
        if have("A", "C", "D", "E"):
            P["S2 accuracy of C, D, E within 3 points of A"] = all(abs(D[f"acc:{k}-A"]["d"]) <= 3 for k in "CDE")
            P["S4 C minus A in the second-turn number at most -10, interval excluding zero"] = D["conf:C-A"]["d"] <= -10 and D["conf:C-A"]["hi"] < 0
            P["S5 D minus E in AUROC between 0 and +0.10"] = 0 <= D["auroc:D-E"]["d"] <= 0.10
            a, b = S["D"]["assertive_minus_other_accuracy"], S["E"]["assertive_minus_other_accuracy"]
            P["S6 in D assertive answers at least 5 points more often right, in E within 5"] = (a is not None and a >= 5) and (b is None or abs(b) <= 5)
            travels = P["S4 C minus A in the second-turn number at most -10, interval excluding zero"] or (
                D["auroc:D-E"]["d"] >= 0.05 and D["auroc:D-E"]["lo"] > 0)
            label_only = abs(D["conf:C-A"]["d"]) <= 5 and D["auroc:D-E"]["d"] < 0.05
            res["reading"] = "certainty travels without a number" if travels else "the risk is the explicit label" if label_only else "mixed"
            print(f"{fam:10s} reading: {res['reading']}")
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        out[fam] = {**res, "predictions": P}
    (RES / "prose_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
