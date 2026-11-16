"""Ask arm (PROTOCOL.md, 2026-11-15 23:51): does the trained confidence level show when the
confidence is asked for in another way?

  python3 loopgain/ask.py     # reads results/runs_ask_<fam>.json, writes results/ask_stats.json
Ways of asking: T (trained format), F1 (reworded, one turn), F2 (answer first, then a second
turn). Carry-over c_f = D_f / D_T, D = stated confidence of the TCAL student minus the SCAL
student, on the questions in format in both cells.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from ext import sel50  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
FAMS = ("qwen35", "granite41")
WAYS = {"T": "eval", "F1": "eval_ask_f1", "F2": "eval_ask_f2"}
MIN_FORMAT = 0.90


def kind(name: str) -> str:
    if name == "base__ask":
        return "untrained"
    lab = "own" if "__base__" in name else name.split("ds41f__")[1].split("__")[0].split("_")[0]
    return lab + ("" if "__s1__" in name else "_s2")


def ok(rows):
    return [r for r in rows if r["answer"] is not None and r["conf"] is not None]


def summary(rows):
    k = ok(rows)
    acc = 100 * sum(bool(r["correct"]) for r in k) / len(k)
    conf = sum(r["conf"] for r in k) / len(k)
    forms: dict = {}
    for r in rows:
        forms[str(r.get("form"))] = forms.get(str(r.get("form")), 0) + 1
    return {"n": len(k), "in_format": len(k) / len(rows), "accuracy": acc, "conf": conf, "gap": conf - acc,
            "auroc": core.auroc([r["conf"] for r in k], [bool(r["correct"]) for r in k]), "sel50": sel50(k),
            "as_asked": forms.get("asked", 0) / len(rows), "as_trained": forms.get("trained", 0) / len(rows),
            "added_conf": sum(bool(r.get("added")) for r in rows) / len(rows)}


def diff(a, b, n=2000, seed=1):
    """Mean confidence of a minus b on questions in format in both, with a paired bootstrap."""
    A, B = {r["id"]: r["conf"] for r in ok(a)}, {r["id"]: r["conf"] for r in ok(b)}
    d = [A[i] - B[i] for i in sorted(set(A) & set(B))]
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {"d": sum(d) / len(d), "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n)], "n": len(d)}


def main():
    out = {}
    for fam in FAMS:
        f = RES / f"runs_ask_{fam}.json"
        if not f.exists():
            continue
        runs = {kind(n): r for n, r in json.loads(f.read_text())["runs"].items()}
        res = {"cells": {}, "carry": {}}
        for rd, sfx in (("sampled", "_sampled"), ("greedy", "")):
            print(f"\n{fam}, {rd} answer")
            print(f"{'cell':10s} " + " | ".join(f"{w:>3s} fmt   acc  conf    gap  auc" for w in WAYS))
            for k in ("tcal", "scal", "shuf", "own", "tcal_s2", "scal_s2", "untrained"):
                if k not in runs:
                    continue
                c = {w: summary(runs[k]["eval"][stem + sfx]) for w, stem in WAYS.items()}
                res["cells"][f"{k}:{rd}"] = c
                print(f"{k:10s} " + " | ".join(
                    f"{c[w]['in_format']:7.3f} {c[w]['accuracy']:5.1f} {c[w]['conf']:5.1f} {c[w]['gap']:+6.1f} "
                    f"{(c[w]['auroc'] or 0):.2f}" for w in WAYS))
            for s in ("", "_s2"):
                if "tcal" + s not in runs or "scal" + s not in runs:
                    continue
                D = {w: diff(runs["tcal" + s]["eval"][stem + sfx], runs["scal" + s]["eval"][stem + sfx])
                     for w, stem in WAYS.items()}
                usable = {w: min(res["cells"][f"{k}{s}:{rd}"][w]["in_format"] for k in ("tcal", "scal")) >= MIN_FORMAT
                          for w in WAYS}
                res["carry"][f"seed{'2' if s else '1'}:{rd}"] = {
                    w: {**D[w], "c": D[w]["d"] / D["T"]["d"], "c_lo": D[w]["lo"] / D["T"]["d"],
                        "c_hi": D[w]["hi"] / D["T"]["d"], "usable": usable[w]} for w in WAYS}
                print(f"  seed {'2' if s else '1'}: " + "; ".join(
                    f"D_{w} {D[w]['d']:+5.1f} ({D[w]['lo']:+5.1f},{D[w]['hi']:+5.1f}) c {D[w]['d'] / D['T']['d']:.2f}"
                    + ("" if usable[w] else " [below 90% in format]") for w in WAYS))
        out[fam] = res
    old = {fam: {c["teacher"]: c for c in json.loads((RES / "dist_cells.json").read_text())
                 if c["family"] == fam and c["seed"] == 1 and c["readout"] == "sampled"} for fam in out}
    print("\npredictions (sampled answer unless stated)")
    for fam, res in out.items():
        C = {k.split(":")[0]: v for k, v in res["cells"].items() if k.endswith(":sampled")}
        cr = res["carry"]
        num = {"tcal": old[fam]["ds41f__tcal"], "scal": old[fam][f"ds41f__scal_{fam}"],
               "shuf": old[fam]["ds41f__shuf"], "own": old[fam]["base"]}
        P = {
            "A1 retrained cells reproduce the T gap within 1.0": all(abs(C[k]["T"]["gap"] - num[k]["gap"]) <= 1.0 for k in num),
            "A2 >= 90% in format under F1 and F2, every cell": all(c[w]["in_format"] >= MIN_FORMAT for c in C.values() for w in ("F1", "F2")),
            "A3 c_F1 >= 0.6": cr["seed1:sampled"]["F1"]["c"] >= 0.6,
            "A4 0.3 <= c_F2 <= c_F1": 0.3 <= cr["seed1:sampled"]["F2"]["c"] <= cr["seed1:sampled"]["F1"]["c"],
            "A5 accuracy under F1, F2 within 4 of T": all(abs(c[w]["accuracy"] - c["T"]["accuracy"]) <= 4 for c in C.values() for w in ("F1", "F2")),
            "A6 F1: tcal gap >= +15, scal |gap| <= 12": C["tcal"]["F1"]["gap"] >= 15 and abs(C["scal"]["F1"]["gap"]) <= 12,
            "A7 F1: AUROC tcal >= 0.62 and +0.08 over shuf": C["tcal"]["F1"]["auroc"] >= 0.62 and C["tcal"]["F1"]["auroc"] - C["shuf"]["F1"]["auroc"] >= 0.08,
            "A8 untrained states >= 85% in all three": all(C["untrained"][w]["conf"] >= 85 for w in WAYS),
            "A9 c differs <= 0.15 between seeds": all(abs(cr["seed1:sampled"][w]["c"] - cr["seed2:sampled"][w]["c"]) <= 0.15 for w in ("F1", "F2")),
        }
        R = {"refuted: c_F1 < 0.3": cr["seed1:sampled"]["F1"]["c"] < 0.3, "refuted: c_F2 < 0.2": cr["seed1:sampled"]["F2"]["c"] < 0.2}
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        for k, v in R.items():
            print(f"{fam:10s} {'YES' if v else 'no '}    {k}")
        res["predictions"], res["refutation"] = P, R
        res["usd"] = sum(r["meta"].get("usd") or 0 for r in json.loads((RES / f"runs_ask_{fam}.json").read_text())["runs"].values())
    (RES / "ask_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
