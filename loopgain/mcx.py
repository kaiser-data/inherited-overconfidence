"""Multiple-choice arm (PROTOCOL.md, 2026-11-16): analysis.

  python3 loopgain/mcx.py     # reads results/runs_mc_<fam>.json, writes results/mc_stats.json
Per cell on ME (1,000 MMLU-Pro questions of our own partition), sampled and greedy answer:
in-format share, accuracy, stated confidence, gap with a question-bootstrap interval, ECE, AUROC,
accuracy on the more confident half. Then the registered predictions M1-M7.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from ext import sel50, summary  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
FAMS = ("qwen35", "granite41")
READ = {"sampled": "eval_sampled", "greedy": "eval"}


def kind(name: str) -> str:
    if name == "base_mc":
        return "untrained"
    return name.split("ds41f__")[1].split("__")[0].split("_")[0] + ("" if "__s1__" in name else "_s2")


def full(rows):
    s = summary(rows)
    ok = [r for r in rows if r["answer"] is not None and r["conf"] is not None]
    s["ece"] = core.ece([r["conf"] for r in ok], [bool(r["correct"]) for r in ok])
    s["accuracy_all"] = 100 * sum(bool(r["correct"]) for r in rows) / len(rows)  # not in format = wrong
    return s


def paired(a, b, stat, n=2000, seed=1):
    """stat(a) - stat(b) on the questions in format in both cells, paired question-bootstrap."""
    A = {r["id"]: r for r in a if r["answer"] is not None and r["conf"] is not None}
    B = {r["id"]: r for r in b if r["answer"] is not None and r["conf"] is not None}
    ids = sorted(set(A) & set(B))
    rng = random.Random(seed)
    d0 = stat([A[i] for i in ids]) - stat([B[i] for i in ids])
    bs = []
    for _ in range(n):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        bs.append(stat([A[i] for i in s]) - stat([B[i] for i in s]))
    bs.sort()
    return {"d": d0, "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n)], "n": len(ids)}


def main():
    out = {}
    au = lambda rs: core.auroc([r["conf"] for r in rs], [bool(r["correct"]) for r in rs]) or 0.5
    for fam in FAMS:
        f = RES / f"runs_mc_{fam}.json"
        if not f.exists():
            continue
        runs = {kind(n): r for n, r in json.loads(f.read_text())["runs"].items()}
        cells = {}
        print(f"\n{fam}")
        print(f"{'cell':10s} {'label':>6s} | " + " | ".join(f"{rd:>7s} fmt   acc  conf    gap (95%)         ece  auc sel50" for rd in READ))
        for k in ("raw", "tcal", "scal", "shuf", "tcal_s2", "scal_s2", "untrained"):
            if k not in runs:
                continue
            m = runs[k]["meta"]
            c = {"label": (m.get("data") or {}).get("train_conf_mean"), "usd": m.get("usd"), "seconds": m.get("seconds"),
                 "n_records": (m.get("train") or {}).get("n_records"), "n_at_max_len": (m.get("train") or {}).get("n_at_max_len"),
                 "checksum_equal": m.get("checksum_equal")}
            for rd, stem in READ.items():
                c[rd] = full(runs[k]["eval"][stem])
            cells[k] = c
            print(f"{k:10s} {(c['label'] or float('nan')):6.1f} | " + " | ".join(
                f"{c[rd]['parsed']:11.3f} {c[rd]['accuracy']:5.1f} {c[rd]['conf']:5.1f} {c[rd]['gap']:+6.1f} "
                f"({c[rd]['gap_lo']:+5.1f},{c[rd]['gap_hi']:+5.1f}) {c[rd]['ece']:5.1f} {c[rd]['auroc']:.2f} {c[rd]['sel50']:5.1f}" for rd in READ))
        res = {"cells": cells}
        if "tcal" in runs and "shuf" in runs:
            res["auroc:tcal-shuf"] = paired(runs["tcal"]["eval"]["eval_sampled"], runs["shuf"]["eval"]["eval_sampled"], au)
            res["sel50:tcal-shuf"] = paired(runs["tcal"]["eval"]["eval_sampled"], runs["shuf"]["eval"]["eval_sampled"], sel50)
            print("  tcal - shuf: AUROC {d:+.3f} ({lo:+.3f},{hi:+.3f})".format(**res["auroc:tcal-shuf"]),
                  " more confident half {d:+.1f} ({lo:+.1f},{hi:+.1f})".format(**res["sel50:tcal-shuf"]))
        out[fam] = res
    info = json.loads((RES / "mc" / "labels.json").read_text())
    tacc = info["sets"]["tcal"]["teacher_accuracy"]
    print("\npredictions (sampled answer unless stated)")
    for fam, res in out.items():
        c = res["cells"]
        s1 = [k for k in ("raw", "tcal", "scal", "shuf") if k in c]
        S = lambda k: c[k]["sampled"]
        P = {}
        P["M1 |C - L| <= 5 in every seed-1 cell"] = all(abs(S(k)["conf"] - c[k]["label"]) <= 5 for k in s1)
        if all(k in c for k in ("tcal", "scal")):
            acc = [S(k)["accuracy"] for k in s1]
            P["M2 accuracy differs <= 4 between label sets; students >= 15 below the teacher"] = (
                max(acc) - min(acc) <= 4 and all(tacc - a >= 15 for a in acc))
            P["M3 tcal gap >= +15"] = S("tcal")["gap"] >= 15
            P["M4 scal |gap| <= 10"] = abs(S("scal")["gap"]) <= 10
            P["M5 greedy: tcal gap >= +15, scal |gap| <= 12"] = c["tcal"]["greedy"]["gap"] >= 15 and abs(c["scal"]["greedy"]["gap"]) <= 12
        if "shuf" in c and "tcal" in c:
            P["M6 shuf: conf within 3 and accuracy within 4 of tcal"] = (
                abs(S("shuf")["conf"] - S("tcal")["conf"]) <= 3 and abs(S("shuf")["accuracy"] - S("tcal")["accuracy"]) <= 4)
            P["M7 AUROC tcal >= 0.60 and >= 0.05 above shuf"] = S("tcal")["auroc"] >= 0.60 and S("tcal")["auroc"] - S("shuf")["auroc"] >= 0.05
        if "tcal_s2" in c and "scal_s2" in c:
            P["M8 seed 2 within 4 points of seed 1 (gap)"] = all(abs(c[k + "_s2"]["sampled"]["gap"] - S(k)["gap"]) <= 4 for k in ("tcal", "scal"))
        P["M9 >= 95% in format in every trained cell"] = all(c[k][rd]["parsed"] >= 0.95 for k in c if k != "untrained" for rd in READ)
        R = {}
        if "tcal" in c and "scal" in c:
            R["refuted: tcal gap interval below +10"] = S("tcal")["gap_hi"] < 10
            R["refuted: scal |gap| > 15"] = abs(S("scal")["gap"]) > 15
            R["refuted: |C - L| > 10 in a cell"] = any(abs(S(k)["conf"] - c[k]["label"]) > 10 for k in s1)
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        for k, v in R.items():
            print(f"{fam:10s} {'YES' if v else 'no '}    {k}")
        res["predictions"], res["refutation"], res["teacher_record_accuracy"] = P, R, tacc
    (RES / "mc_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
