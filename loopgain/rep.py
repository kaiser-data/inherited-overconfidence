"""Seeds for the shuffled control of Result 3 (PROTOCOL.md, 2026-11-16 03:37): analysis.

  python3 loopgain/rep.py     # reads results/runs_dist_<fam>.json and runs_rep_<fam>.json, writes results/rep_stats.json
Per student and training seed: the calibrated-teacher student against the student trained on the same
values shuffled across records; AUROC and accuracy on the more confident half, paired on the test
questions (sampled answer). Then the registered predictions R1-R5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import core  # noqa: E402
from ext import sel50, summary  # noqa: E402
from mcx import paired  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
T = config.DIST_SEED_TEACHER


def au(rs):
    return core.auroc([r["conf"] for r in rs], [bool(r["correct"]) for r in rs]) or 0.5


def main():
    out = {}
    label = json.loads((RES / "dist" / "labels.json").read_text())[T]["sets"][f"{T}__shuf"]["label_mean"]
    for fam in config.NEW_FAMS:
        runs = dict(json.loads((RES / f"runs_dist_{fam}.json").read_text())["runs"])
        f = RES / f"runs_rep_{fam}.json"
        if f.exists():
            runs.update(json.loads(f.read_text())["runs"])
        cell = lambda lab, s: next((r for r in runs.values() if r["meta"]["teacher"] == f"{T}__{lab}" and r["meta"]["seed"] == s), None)
        seeds, P = {}, {}
        print(f"\n{fam}\nseed | tcal: auc half gap | shuf: fmt  conf  auc half gap | auc diff (95%)        | half diff (95%)")
        for s in (1, 2, 3):
            a, b = cell("tcal", s), cell("shuf", s)
            if not a or not b:
                continue
            A, B = a["eval"]["eval_sampled"], b["eval"]["eval_sampled"]
            sa, sb = summary(A), summary(B)
            d, h = paired(A, B, au), paired(A, B, sel50)
            seeds[str(s)] = {"tcal": sa, "shuf": sb, "auroc_diff": d, "half_diff": h, "usd": b["meta"].get("usd"),
                             "checksum_equal": b["meta"].get("checksum_equal")}
            print(f"{s:4d} | {sa['auroc']:.2f} {sa['sel50']:5.1f} {sa['gap']:+5.1f} | {sb['parsed']:.3f} {sb['conf']:5.1f} {sb['auroc']:.2f} {sb['sel50']:5.1f} {sb['gap']:+5.1f} | "
                  f"{d['d']:+.3f} ({d['lo']:+.3f},{d['hi']:+.3f}) | {h['d']:+5.1f} ({h['lo']:+5.1f},{h['hi']:+5.1f})")
        new = [seeds[s] for s in ("2", "3") if s in seeds]
        if new and "1" in seeds:
            P["R1 AUROC TCAL - SHUF >= 0.10 in seeds 2 and 3"] = all(e["auroc_diff"]["d"] >= 0.10 for e in new)
            P["R2 AUROC SHUF <= 0.60 and TCAL >= 0.65"] = all(e["shuf"]["auroc"] <= 0.60 and e["tcal"]["auroc"] >= 0.65 for e in new)
            P["R3 more confident half TCAL - SHUF >= 8 points"] = all(e["half_diff"]["d"] >= 8 for e in new)
            P["R4 SHUF conf within 3 of its label, gap within 4 of seed 1"] = all(
                abs(e["shuf"]["conf"] - label) <= 3 and abs(e["shuf"]["gap"] - seeds["1"]["shuf"]["gap"]) <= 4 for e in new)
            P["R5 >= 95% in format"] = all(e["shuf"]["parsed"] >= 0.95 for e in new)
            R = {"refuted: AUROC difference below 0.05 in a seed": any(e["auroc_diff"]["d"] < 0.05 for e in new)}
            for k, v in P.items():
                print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
            for k, v in R.items():
                print(f"{fam:10s} {'YES' if v else 'no '}    {k}")
            out[fam] = {"seeds": seeds, "predictions": P, "refutation": R, "label_mean_shuf": label}
        else:
            out[fam] = {"seeds": seeds}
    (RES / "rep_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
