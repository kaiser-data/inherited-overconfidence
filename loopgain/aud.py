"""Audit-size arm (PROTOCOL.md, 2026-11-16): how many checked student answers does the
student-fitted label need?

  python3 aud.py build     # label sets from n checked answers -> results/dist/ds41f__scal<n>_<fam>/
  python3 aud.py report    # table from results/runs_aud_<fam>.json -> results/aud_stats.json

SCAL fits the student's correctness on the teacher's stated confidence using all 1,978 records.
Here the fit sees only the first n records of the fixed audit order (nested), and is applied to
all records. `build` also repeats the fit on 500 random draws per n (CPU only) to show how much
the resulting mean label depends on which n answers were checked.
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
import distill  # noqa: E402

RES = HERE.parent / "results"
T = config.DIST_SEED_TEACHER
DRAWS = 500


def fit_labels(rows, sc, ids) -> dict[str, int]:
    sub = [r for r in rows if r["id"] in ids]
    blocks = core.isotonic_fit([r["conf"] for r in sub], [1.0 if sc.get(r["id"], False) else 0.0 for r in sub])
    return {r["id"]: core.conf5(core.isotonic_predict(blocks, r["conf"])) for r in rows}


def build():
    _, perm = distill.load_d()
    rows = distill.retained(distill.teacher_rows(T))
    kept = {r["id"] for r in rows}
    order = [i for i in perm if i in kept]
    out = {}
    for fam, src in distill.STUDENTS.items():
        sc = {r["id"]: bool(r["correct"]) for r in map(json.loads, src.read_text().splitlines())}
        base_acc = 100 * sum(sc.get(i, False) for i in kept) / len(kept)
        full = fit_labels(rows, sc, kept)
        out[fam] = {"student_base_accuracy": base_acc, "full_label_mean": sum(full.values()) / len(full), "sizes": {}}
        for n in config.AUD_SIZES:
            labels = fit_labels(rows, sc, set(order[:n]))
            recs = distill.relabel(rows, labels)
            st = distill.stats(recs, sc)
            assert st["answers_sha"] == distill.answers_hash(rows)
            d = distill.OUT / f"{T}__scal{n}_{fam}"
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "d_main.jsonl", "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            rng = random.Random(config.DATA_SEED + n)
            means = sorted(sum(fit_labels(rows, sc, set(rng.sample(order, n))).values()) / len(rows) for _ in range(DRAWS))
            q = lambda p: means[int(p * (DRAWS - 1))]
            out[fam]["sizes"][n] = {"label_mean": st["label_mean"], "levels": len(st["hist"]), "hist": st["hist"],
                                    "draws_lo": q(0.025), "draws_med": q(0.5), "draws_hi": q(0.975),
                                    "draws_within5": sum(abs(m - base_acc) <= 5 for m in means) / DRAWS}
            print(f"{fam:10s} n {n:4d}  label mean {st['label_mean']:5.1f} ({len(st['hist'])} levels)  student-base acc {base_acc:5.1f}  "
                  f"random draws: median {q(0.5):5.1f}, 95% {q(0.025):5.1f} to {q(0.975):5.1f}, within 5 points of accuracy {out[fam]['sizes'][n]['draws_within5']:.0%}")
        print(f"{fam:10s} n {len(kept)} (SCAL) label mean {out[fam]['full_label_mean']:5.1f}")
    (RES / "aud_labels.json").write_text(json.dumps(out, indent=1))


def report():
    from ext import summary
    labels = json.loads((RES / "aud_labels.json").read_text())
    ask = json.loads((RES / "ask_stats.json").read_text())
    out = {}
    for fam in config.NEW_FAMS:
        f = RES / f"runs_aud_{fam}.json"
        if not f.exists():
            continue
        runs = json.loads(f.read_text())["runs"]
        cells = {}
        print(f"\n{fam}   (sampled answer; last row: all records, from the ask arm)")
        print(f"{'checked':>8s} {'label':>6s} {'fmt':>6s} {'acc':>6s} {'conf':>6s} {'gap (95%)':>22s} {'auc':>5s} {'sel50':>6s} | greedy gap")
        for n in config.AUD_SIZES:
            name = next(k for k in runs if f"__scal{n}_{fam}__" in k)
            s, g = summary(runs[name]["eval"]["eval_sampled"]), summary(runs[name]["eval"]["eval"])
            m = runs[name]["meta"]
            cells[n] = {"label": m["data"]["train_conf_mean"], "sampled": s, "greedy": g, "usd": m.get("usd"),
                        "checksum_equal": m.get("checksum_equal")}
            print(f"{n:8d} {cells[n]['label']:6.1f} {s['parsed']:6.3f} {s['accuracy']:6.1f} {s['conf']:6.1f} "
                  f"{s['gap']:+7.1f} ({s['gap_lo']:+5.1f},{s['gap_hi']:+5.1f}) {s['auroc']:5.2f} {s['sel50']:6.1f} | {g['gap']:+6.1f}")
        a = ask[fam]["cells"]["scal:sampled"]["T"]
        print(f"{'1978':>8s} {labels[fam]['full_label_mean']:6.1f} {a['in_format']:6.3f} {a['accuracy']:6.1f} {a['conf']:6.1f} {a['gap']:+7.1f} {'':15s} {a['auroc']:5.2f} {a['sel50']:6.1f}")
        fc = {n: cells[n]["label"] - a["accuracy"] for n in cells}  # forecast: label minus accuracy of the all-records cell
        P = {
            "U1 |C - L| <= 5 in every cell": all(abs(c["sampled"]["conf"] - c["label"]) <= 5 for c in cells.values()),
            "U2 gap within 5 of the forecast (label minus accuracy) in every cell": all(abs(cells[n]["sampled"]["gap"] - fc[n]) <= 5 for n in cells),
            "U3 |gap| <= 10 with 250 and 500 checked": all(abs(cells[n]["sampled"]["gap"]) <= 10 for n in (250, 500)),
            "U4 accuracy within 3 of the all-records cell": all(abs(c["sampled"]["accuracy"] - a["accuracy"]) <= 3 for c in cells.values()),
            "U5 AUROC >= 0.62 with 250 and 500 checked": all(cells[n]["sampled"]["auroc"] >= 0.62 for n in (250, 500)),
        }
        for k, v in P.items():
            print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
        out[fam] = {"cells": cells, "all_records": a, "predictions": P}
    (RES / "aud_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else report()
