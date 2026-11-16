"""Level sweep (PROTOCOL.md, 2026-11-16 06:0x): can the stated confidence be dialled to chosen levels?

  python3 lvl.py build     # label sets with mean 50, 65, 80 -> results/dist/ds41f__lvl<t>/
  python3 lvl.py report    # reads results/runs_lvl_<fam>.json (+ the 90 and 98 cells of the lead experiment)
The teacher-calibrated labels (mean 90) are moved by one shift in log-odds, the same for every record, so
the order of the records is kept; every question and answer is unchanged. Written labels are multiples of 5.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import core  # noqa: E402
import distill  # noqa: E402

RES = HERE.parent / "results"
T = config.DIST_SEED_TEACHER
TARGETS = config.LVL_TARGETS
N_BOOT = 2000


def moved(label: float, b: float) -> int:
    """One label under the log-odds shift b, as written into a record (multiple of 5, never below 5)."""
    p = min(0.98, max(0.02, label / 100))
    return max(5, core.conf5(1 / (1 + math.exp(-(math.log(p / (1 - p)) + b)))))


def shift_for(labels: list[float], target: float) -> float:
    """The shift whose written labels have the mean closest to the target (the mean is monotone in b)."""
    mean = lambda b: sum(moved(x, b) for x in labels) / len(labels)
    lo, hi = -8.0, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(mid) < target else (lo, mid)
    return min((lo, hi), key=lambda b: abs(mean(b) - target))


def build():
    rows = [json.loads(l) for l in (distill.OUT / f"{T}__tcal" / "d_main.jsonl").open()]
    base = [float(r["conf"]) for r in rows]
    out = {}
    for t in TARGETS:
        b = shift_for(base, t)
        recs = [{**r, "conf": moved(float(r["conf"]), b)} for r in rows]
        assert distill.answers_hash(recs) == distill.answers_hash(rows)
        assert all((x <= y) == (moved(x, b) <= moved(y, b)) or moved(x, b) == moved(y, b) for x, y in zip(base, base[1:]))
        st = distill.stats(recs)
        d = distill.OUT / f"{T}__lvl{t}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "d_main.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[str(t)] = {"shift": b, "label_mean": st["label_mean"], "hist": st["hist"], "levels": len(st["hist"])}
        print(f"target {t}: shift {b:+.2f}, label mean {st['label_mean']:.1f}, labels {st['hist']}")
    (RES / "lvl_labels.json").write_text(json.dumps(out, indent=1))


def _ok(rows):
    return [r for r in rows if r.get("answer") is not None and r.get("conf") is not None]


def point(rows: list[dict]) -> dict:
    ok = _ok(rows)
    acc, conf = 100 * sum(bool(r["correct"]) for r in ok) / len(ok), sum(r["conf"] for r in ok) / len(ok)
    return {"parsed": len(ok) / len(rows), "accuracy": acc, "conf": conf, "gap": conf - acc,
            "auroc": core.auroc([r["conf"] for r in ok], [bool(r["correct"]) for r in ok])}


def paired_acc(a: list[dict], b: list[dict]) -> dict:
    """Accuracy of a minus b on the questions both answered in format; question bootstrap."""
    B = {r["id"]: r for r in _ok(b)}
    pairs = [(bool(r["correct"]), bool(B[r["id"]]["correct"])) for r in _ok(a) if r["id"] in B]
    d = lambda ps: 100 * (sum(x for x, _ in ps) - sum(y for _, y in ps)) / len(ps)
    rng = random.Random(1)
    bs = sorted(d([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(N_BOOT))
    return {"d": d(pairs), "lo": bs[int(0.025 * N_BOOT)], "hi": bs[int(0.975 * N_BOOT)], "n": len(pairs)}


def slope(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def report():
    out = {}
    for fam in config.NEW_FAMS:
        f = RES / f"runs_lvl_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        lead = json.loads((RES / f"runs_dist_{fam}.json").read_text())["runs"]
        ref = next(r for r in lead.values() if r["meta"]["teacher"] == f"{T}__tcal" and r["meta"]["seed"] == 1)
        raw = next(r for r in lead.values() if r["meta"]["teacher"] == f"{T}__raw" and r["meta"]["seed"] == 1)
        cells = {str(t): next(r for r in runs.values() if r["meta"]["teacher"] == f"{T}__lvl{t}") for t in TARGETS
                 if any(r["meta"]["teacher"] == f"{T}__lvl{t}" for r in runs.values())}
        cells["90"], res = ref, {}
        print(f"\n===== {fam}")
        for rd, key in (("sampled", "eval_sampled"), ("greedy", "eval")):
            pts = {}
            for k, c in sorted(cells.items(), key=lambda kv: int(kv[0])):
                pts[k] = {**point(c["eval"][key]), "label": c["meta"]["data"]["train_conf_mean"],
                          "acc_vs_90": None if k == "90" else paired_acc(c["eval"][key], ref["eval"][key])}
            xs = [p["label"] for p in pts.values()]
            sl = {"conf_on_label": slope(xs, [p["conf"] for p in pts.values()]),
                  "accuracy_on_label": slope(xs, [p["accuracy"] for p in pts.values()])}
            res[rd] = {"points": pts, "slopes": sl,
                       "teachers_own": {**point(raw["eval"][key]), "label": raw["meta"]["data"]["train_conf_mean"]}}
            print(f"{rd}: slope of says on label {sl['conf_on_label']:.2f}, of is-right on label {sl['accuracy_on_label']:+.3f}")
            for k, p in pts.items():
                d = p["acc_vs_90"]
                print(f"  label {p['label']:5.1f}: says {p['conf']:5.1f} ({p['conf'] - p['label']:+.1f}), right {p['accuracy']:5.1f}, AUROC {p['auroc']:.2f}, "
                      f"in format {100 * p['parsed']:.1f}%" + ("" if d is None else f", right minus the 90 cell {d['d']:+.1f} ({d['lo']:+.1f},{d['hi']:+.1f})"))
        new = [k for k in res["sampled"]["points"] if k != "90"]
        P = {"V1 says within 3 points of the label mean at every new level, both readouts": all(
                abs(res[rd]["points"][k]["conf"] - res[rd]["points"][k]["label"]) <= 3 for rd in res for k in new),
             "V2 slope of says on label between 0.9 and 1.1, both readouts": all(0.9 <= res[rd]["slopes"]["conf_on_label"] <= 1.1 for rd in res),
             "V3 greedy accuracy equivalent to the 90 cell at every new level (difference within 3, interval within 5)": all(
                abs(res["greedy"]["points"][k]["acc_vs_90"]["d"]) <= 3 and -5 <= res["greedy"]["points"][k]["acc_vs_90"]["lo"]
                and res["greedy"]["points"][k]["acc_vs_90"]["hi"] <= 5 for k in new),
             "V4 sampled accuracy within 4 points of the 90 cell at every new level": all(
                abs(res["sampled"]["points"][k]["acc_vs_90"]["d"]) <= 4 for k in new),
             "V5 AUROC at least 0.62 at every new level (sampled)": all(res["sampled"]["points"][k]["auroc"] >= 0.62 for k in new)}
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        out[fam] = {**res, "predictions": P, "levels_run": new}
    (RES / "lvl_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
