"""Extension arm (PROTOCOL.md, 2026-11-15 21:45): wording-only confidence and a second question set.

  python3 loopgain/ext.py        # reads results/runs_ext_<fam>.json, writes results/ext_stats.json
Per cell and question set (E = TriviaQA test questions, E2 = Natural Questions), sampled and greedy:
accuracy, stated confidence, gap with a question-bootstrap interval, AUROC, accuracy on the more
confident half. Then the registered predictions W1-W5 and X1-X5.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
FAMS = ("qwen35", "granite41")
READ = {"E": ("eval_sampled", "eval"), "E2": ("eval2_sampled", "eval2")}


def sel50(rs):
    k, got, by = len(rs) / 2, 0.0, {}
    for r in rs:
        c = by.setdefault(r["conf"], [0, 0])
        c[0] += 1
        c[1] += bool(r["correct"])
    for conf in sorted(by, reverse=True):
        m, right = by[conf]
        take = min(m, k)
        got += right * take / m
        k -= take
        if k <= 0:
            break
    return 100 * got / (len(rs) / 2)


def summary(rows, n=2000):
    ok = [r for r in rows if r["answer"] is not None and r["conf"] is not None]
    acc = lambda rs: 100 * sum(bool(r["correct"]) for r in rs) / len(rs)
    conf = lambda rs: sum(r["conf"] for r in rs) / len(rs)
    rng = random.Random(1)
    gaps = sorted(conf(s) - acc(s) for s in ([ok[rng.randrange(len(ok))] for _ in ok] for _ in range(n)))
    return {"n": len(ok), "parsed": len(ok) / len(rows), "accuracy": acc(ok), "conf": conf(ok), "gap": conf(ok) - acc(ok),
            "gap_lo": gaps[int(0.025 * n)], "gap_hi": gaps[int(0.975 * n)],
            "auroc": core.auroc([r["conf"] for r in ok], [bool(r["correct"]) for r in ok]), "sel50": sel50(ok),
            "levels": len({r["conf"] for r in ok})}


def kind(name):
    lab = "own" if "__base__" in name else name.split("ds41f__")[1].split("__")[0].split("_")[0]
    return lab + ("_w" if name.endswith("__w") else "")


def main():
    out = {}
    for fam in FAMS:
        f = RES / f"runs_ext_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        cells = {}
        for name, r in runs.items():
            c = {"label": r["meta"]["data"]["train_conf_mean"], "usd": r["meta"].get("usd"), "seconds": r["meta"].get("seconds")}
            for qs, (smp, grd) in READ.items():
                if smp in r["eval"]:
                    c[qs] = summary(r["eval"][smp])
                    c[qs + "_greedy"] = summary(r["eval"][grd])
            cells[kind(name)] = c
        out[fam] = cells
        print(f"\n{fam}")
        print(f"{'cell':8s} {'label':>6s} | " + " | ".join(f"{q:>4s} acc  conf   gap (95%)        auc  sel50 parsed" for q in ("E", "E2")))
        for k in ("tcal_w", "scal_w", "tcal", "scal", "shuf", "own"):
            if k not in cells:
                continue
            c = cells[k]
            print(f"{k:8s} {c['label']:6.1f} | " + " | ".join(
                f"{c[q]['accuracy']:8.1f} {c[q]['conf']:5.1f} {c[q]['gap']:+6.1f} ({c[q]['gap_lo']:+5.1f},{c[q]['gap_hi']:+5.1f}) "
                f"{c[q]['auroc']:.2f} {c[q]['sel50']:6.1f} {c[q]['parsed']:6.3f}" for q in ("E", "E2") if q in c))
    (RES / "ext_stats.json").write_text(json.dumps(out, indent=1))
    old = {fam: {c["teacher"]: c for c in json.loads((RES / "dist_cells.json").read_text())
                 if c["family"] == fam and c["seed"] == 1 and c["readout"] == "sampled"} for fam in out}
    print("\npredictions (sampled readout)")
    for fam, c in out.items():
        o = old[fam]
        num = {"tcal": o["ds41f__tcal"], "scal": o[f"ds41f__scal_{fam}"], "shuf": o["ds41f__shuf"], "own": o["base"]}
        P = {}
        if "tcal_w" in c and "scal_w" in c:
            w = [c["tcal_w"], c["scal_w"]]
            P["W1 parsed >= 97%"] = all(x["E"]["parsed"] >= 0.97 for x in w)
            P["W2 |C-L| <= 5"] = all(abs(x["E"]["conf"] - x["label"]) <= 5 for x in w)
            P["W3 tcal gap >= +20, scal |gap| <= 10"] = c["tcal_w"]["E"]["gap"] >= 20 and abs(c["scal_w"]["E"]["gap"]) <= 10
            P["W4 accuracy within 3 of numeric"] = all(abs(c[k + "_w"]["E"]["accuracy"] - num[k]["accuracy"]) <= 3 for k in ("tcal", "scal"))
            P["W5 tcal AUROC >= 0.62"] = c["tcal_w"]["E"]["auroc"] >= 0.62
        if all(k in c for k in ("tcal", "scal", "shuf", "own")):
            P["X1 reproduces E gap within 1.0"] = all(abs(c[k]["E"]["gap"] - num[k]["gap"]) <= 1.0 for k in num)
            P["X2 conf on E2 within 8 of E"] = all(abs(c[k]["E2"]["conf"] - c[k]["E"]["conf"]) <= 8 for k in num)
            P["X3 E2 accuracy tcal vs scal <= 3"] = abs(c["tcal"]["E2"]["accuracy"] - c["scal"]["E2"]["accuracy"]) <= 3
            P["X4 gaps higher on E2; scal E2 gap > +5"] = all(c[k]["E2"]["gap"] > c[k]["E"]["gap"] for k in num) and c["scal"]["E2"]["gap"] > 5
            P["X5 E2: AUROC tcal >= 0.62, +0.08 over shuf, sel50 +8"] = (
                c["tcal"]["E2"]["auroc"] >= 0.62 and c["tcal"]["E2"]["auroc"] - c["shuf"]["E2"]["auroc"] >= 0.08
                and c["tcal"]["E2"]["sel50"] - c["shuf"]["E2"]["sel50"] >= 8)
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        out[fam]["predictions"] = P
    (RES / "ext_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
