"""Why the fitted label went stale on multiple choice (PROTOCOL.md, 2026-11-16 03:20 and 03:30).

  python3 loopgain/mcl.py build     # own-records label sets -> results/dist/mc_own_<fam>/d_main.jsonl
  python3 loopgain/mcl.py lfit      # the shifted label set of 03:20 (built, never trained on)
  python3 loopgain/mcl.py           # analysis -> results/mcl_stats.json

Own-records control: each student trained on its own untrained answers to the record questions,
confidence fitted to its own correctness. Analyses without new model output:
  M  matched comparison on the evaluation questions ME (untrained with / without reply prefix; trained)
  D  the fitted student's gap split by the identity of the paper's Section 3
  K  cost of the correction after training (n checked answers of the trained student)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import core  # noqa: E402
import distill  # noqa: E402
from ext import summary  # noqa: E402
from mcrecal import ok  # noqa: E402
from mcx import kind, paired  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results"
MCD = RES / "mc"
FAMS, T = config.NEW_FAMS, config.DIST_SEED_TEACHER
SIZES, DRAWS = (50, 100, 250, 500), 500


def _rows(f: Path) -> list[dict]:
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def _write(f: Path, rows: list[dict]):
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def acc(rows: list[dict]) -> float:
    return 100 * sum(bool(r["correct"]) for r in rows) / len(rows)


def conf(rows: list[dict]) -> float:
    return sum(r["conf"] for r in rows) / len(rows)


def halves(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by = {r["id"]: r for r in rows}
    perm = core.seeded_perm(sorted(by), 1)
    return [by[i] for i in perm[:len(perm) // 2]], [by[i] for i in perm[len(perm) // 2:]]


def shift(rows: list[dict]) -> float:
    """Accuracy minus mean stated confidence of the replies in format."""
    a = [r for r in rows if ok(r)]
    return acc(a) - conf(a)


def shifted(label: float, d: float) -> int:
    return int(min(100, max(0, round(label + d))))


def mc_cells_of(fam: str) -> dict:
    runs = json.loads((RES / f"runs_mc_{fam}.json").read_text())["runs"]
    return {kind(n): r["eval"] for n, r in runs.items()}


# ---------------------------------------------------------------- label sets

def build_lfit(fam: str = "qwen35"):
    """SCAL label + D, D from half a of the trained SCAL student's answers (03:20). Not trained on."""
    a, _ = halves(mc_cells_of(fam)["scal"]["eval_sampled"])
    d = shift(a)
    old = _rows(distill.OUT / f"mc_{T}__scal_{fam}" / "d_main.jsonl")
    new = [dict(r, conf=shifted(r["conf"], d)) for r in old]
    assert distill.answers_hash(new) == distill.answers_hash(old)
    _write(distill.OUT / f"mc_{T}__lfit_{fam}" / "d_main.jsonl", new)
    hist: dict = {}
    for r in new:
        hist[r["conf"]] = hist.get(r["conf"], 0) + 1
    info = {"shift": d, "n_calibration": len(a), "n": len(new), "label_mean_scal": conf(old), "label_mean_lfit": conf(new),
            "hist": {str(k): hist[k] for k in sorted(hist)}, "answers_sha": distill.answers_hash(new)}
    (MCD / "lfit_labels.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def own_records(base_rows: list[dict], ids: list[str]) -> list[dict]:
    """The student's own answers to the given record questions, the confidence fitted to its own
    correctness on its own stated confidence (distill.scal on its own rows)."""
    by = {r["id"]: r for r in base_rows}
    rows = [{"id": i, "answer": core.letter(by[i]["answer"]), "conf": by[i]["conf"], "correct": bool(by[i]["correct"])}
            for i in ids if i in by and core.letter(by[i]["answer"]) is not None and by[i]["conf"] is not None]
    labels = distill.scal(rows, {r["id"]: r["correct"] for r in rows})
    return distill.relabel(rows, labels)


def build():
    info = {}
    for fam in FAMS:
        ids = [r["id"] for r in _rows(distill.OUT / f"mc_{T}__scal_{fam}" / "d_main.jsonl")]
        out = own_records(_rows(MCD / "_base" / f"{fam}_d_main.jsonl"), ids)
        _write(distill.OUT / f"mc_own_{fam}" / "d_main.jsonl", out)
        hist: dict = {}
        for r in out:
            hist[r["conf"]] = hist.get(r["conf"], 0) + 1
        info[fam] = {"n": len(out), "n_teacher_records": len(ids), "label_mean": conf(out), "own_accuracy": acc(out),
                     "own_stated_mean": sum(r["teacher_conf"] for r in out) / len(out),
                     "hist": {str(k): hist[k] for k in sorted(hist)}, "answers_sha": distill.answers_hash(out)}
        print(fam, json.dumps(info[fam]))
    (MCD / "own_labels.json").write_text(json.dumps(info, indent=1))


# ---------------------------------------------------------------- analysis

def part(rows: list[dict]) -> dict:
    s = summary(rows)
    o = [r for r in rows if ok(r)]
    s["ece"] = core.ece([r["conf"] for r in o], [bool(r["correct"]) for r in o])
    return s


def cost_of_correction(rows: list[dict], n: int, draws: int = DRAWS) -> dict:
    """n answers of the trained student drawn at random are checked; the correction is applied to the
    other answers. Share of draws whose remaining gap is within 5 points: isotonic fit and constant shift."""
    rows = [r for r in rows if ok(r)]
    rng, iso, shf = random.Random(n), [], []
    for _ in range(draws):
        order = rows[:]
        rng.shuffle(order)
        cal, rest = order[:n], order[n:]
        blocks = core.isotonic_fit([r["conf"] for r in cal], [1.0 if r["correct"] else 0.0 for r in cal])
        iso.append(sum(100 * core.isotonic_predict(blocks, r["conf"]) for r in rest) / len(rest) - acc(rest))
        shf.append(conf(rest) + shift(cal) - acc(rest))
    q = lambda v: {"within5": 100 * sum(abs(x) <= 5 for x in v) / len(v), "lo": sorted(v)[int(0.025 * len(v))],
                   "hi": sorted(v)[int(0.975 * len(v))], "median_abs": sorted(abs(x) for x in v)[len(v) // 2]}
    return {"isotonic": q(iso), "shift": q(shf)}


def main():
    out = {}
    labels = json.loads((MCD / "labels.json").read_text())["sets"]
    own_info = json.loads((MCD / "own_labels.json").read_text()) if (MCD / "own_labels.json").exists() else {}
    for fam in FAMS:
        if not (RES / f"runs_mc_{fam}.json").exists():
            continue
        cells = mc_cells_of(fam)
        base = {rd: _rows(MCD / "_base" / f"{fam}_e_{rd}.jsonl") for rd in ("main", "greedy")}  # untrained, reply prefix
        scal_s, tcal_s = cells["scal"]["eval_sampled"], cells["tcal"]["eval_sampled"]
        res: dict = {}
        f = RES / f"runs_mcl_{fam}.json"
        runs = json.loads(f.read_text())["runs"] if f.exists() else {}
        print(f"\n===== {fam}")
        # M: matched comparison on the evaluation questions
        m = {"untrained_prefix": {"sampled": part(base["main"]), "greedy": part(base["greedy"])},
             "trained_minus_untrained": {"sampled": paired(scal_s, base["main"], acc), "greedy": paired(cells["scal"]["eval"], base["greedy"], acc)}}
        if "base_mcnp" in runs:
            ev = runs["base_mcnp"]["eval"]
            m["untrained_noprefix"] = {"sampled": part(ev["eval_sampled"]), "greedy": part(ev["eval"])}
            m["noprefix_minus_prefix"] = {"sampled": paired(ev["eval_sampled"], base["main"], acc), "greedy": paired(ev["eval"], base["greedy"], acc)}
            for rd in ("greedy", "sampled"):
                e, d = m["untrained_noprefix"][rd], m["noprefix_minus_prefix"][rd]
                print(f"M1 {rd:7s} untrained without prefix: {100 * e['parsed']:.1f}% in format, {e['accuracy']:.1f}% right; "
                      f"minus with prefix on the same {d['n']} questions {d['d']:+.1f} ({d['lo']:+.1f},{d['hi']:+.1f})")
        for rd in ("sampled", "greedy"):
            d = m["trained_minus_untrained"][rd]
            print(f"M2 {rd:7s} untrained with prefix {m['untrained_prefix'][rd]['accuracy']:.1f}; SCAL student minus untrained {d['d']:+.1f} ({d['lo']:+.1f},{d['hi']:+.1f}) n {d['n']}")
        res["matched"] = m
        # D: the identity of Section 3, reference = the model the label was fitted on
        S, Tc = part(scal_s), part(tcal_s)
        ls, lt = labels[f"scal_{fam}"], labels["tcal"]
        a_fit, a_untr = ls["student_base_accuracy"], m["untrained_prefix"]["sampled"]["accuracy"]
        res["decomposition"] = {
            "scal": {"gap": S["gap"], "copying": S["conf"] - ls["label_mean"], "label_gap": ls["label_mean"] - a_fit,
                     "question_sets": a_fit - a_untr, "training": a_untr - S["accuracy"]},
            "tcal": {"gap": Tc["gap"], "copying": Tc["conf"] - lt["label_mean"], "label_gap": lt["label_mean"] - lt["teacher_accuracy"],
                     "accuracy_difference": lt["teacher_accuracy"] - Tc["accuracy"]}}
        for k, d in res["decomposition"].items():
            parts = {p: v for p, v in d.items() if p != "gap"}
            assert abs(sum(parts.values()) - d["gap"]) < 1e-6, (fam, k)
            print(f"D  {k}: gap {d['gap']:+.1f} = " + " ".join(f"{p} {v:+.1f}" for p, v in parts.items()))
        # K: cost of the correction after training
        res["cost"] = {k: {str(n): cost_of_correction(rows, n) for n in SIZES} for k, rows in (("scal", scal_s), ("tcal", tcal_s))}
        for k, d in res["cost"].items():
            print(f"K  {k}: within 5 points, isotonic / shift: " + ", ".join(
                f"n={n} {d[str(n)]['isotonic']['within5']:.0f}% / {d[str(n)]['shift']['within5']:.0f}%" for n in SIZES))
        if fam == "qwen35":  # what the label of 03:20 would give, by arithmetic (never trained)
            a, b = halves(scal_s)
            res["lfit_forecast"] = {"shift": shift(a), "gap_half_b": conf([r for r in b if ok(r)]) + shift(a) - acc([r for r in b if ok(r)])}
            print(f"   label shifted by {res['lfit_forecast']['shift']:+.2f}: arithmetic gap on half b {res['lfit_forecast']['gap_half_b']:+.1f} (not trained)")
        # own-records control
        own = next((r for n, r in runs.items() if "mc_own" in n), None)
        if own:
            O, U, L = part(own["eval"]["eval_sampled"]), m["untrained_prefix"]["sampled"], own_info[fam]["label_mean"]
            Og = part(own["eval"]["eval"])
            res["own"] = {"sampled": O, "greedy": Og, "label": L, "usd": own["meta"].get("usd"), "checksum_equal": own["meta"].get("checksum_equal"),
                          "minus_scal": paired(own["eval"]["eval_sampled"], scal_s, acc), "minus_untrained": paired(own["eval"]["eval_sampled"], base["main"], acc)}
            print(f"own records: label {L:.1f} | sampled fmt {O['parsed']:.3f} acc {O['accuracy']:.1f} conf {O['conf']:.1f} gap {O['gap']:+.1f} ({O['gap_lo']:+.1f},{O['gap_hi']:+.1f}) "
                  f"ece {O['ece']:.1f} auc {O['auroc']:.2f} | greedy acc {Og['accuracy']:.1f} gap {Og['gap']:+.1f}")
            print("   minus SCAL student {d:+.1f} ({lo:+.1f},{hi:+.1f})".format(**res["own"]["minus_scal"]),
                  " minus untrained {d:+.1f} ({lo:+.1f},{hi:+.1f})".format(**res["own"]["minus_untrained"]))
            P = {"O1 accuracy at least 6 below the SCAL student": S["accuracy"] - O["accuracy"] >= 6,
                 "O2 accuracy within 6 of the untrained model with prefix": abs(O["accuracy"] - U["accuracy"]) <= 6,
                 "O3 |C - L| <= 5": abs(O["conf"] - L) <= 5,
                 "O4 |gap| <= 8 and smaller than the SCAL student's": abs(O["gap"]) <= 8 and abs(O["gap"]) < abs(S["gap"]),
                 "O5 >= 95% in format": min(O["parsed"], Og["parsed"]) >= 0.95}
            R = {"refuted: own-records accuracy within 4 of the SCAL student's": abs(S["accuracy"] - O["accuracy"]) <= 4}
            for k, v in P.items():
                print(f"{fam:10s} {'held  ' if v else 'MISSED'} {k}")
            for k, v in R.items():
                print(f"{fam:10s} {'YES' if v else 'no '}    {k}")
            res["predictions"], res["refutation"] = P, R
        E = {}
        if "noprefix_minus_prefix" in m:
            E["M1 prefix changes the untrained model's accuracy by at most 4"] = all(abs(m["noprefix_minus_prefix"][rd]["d"]) <= 4 for rd in ("sampled", "greedy"))
        E["M2 trained minus untrained: both intervals exclude zero"] = all(m["trained_minus_untrained"][rd]["lo"] > 0 for rd in ("sampled", "greedy"))
        c = res["cost"]["scal"]
        E["K isotonic and shift within 5 percentage points from n = 250"] = all(
            abs(c[str(n)]["isotonic"]["within5"] - c[str(n)]["shift"]["within5"]) <= 5 for n in (250, 500))
        for k, v in E.items():
            print(f"{fam:10s} {'as expected' if v else 'NOT as expected'}  {k}")
        res["expectations"] = E
        out[fam] = res
    (RES / "mcl_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "lfit": build_lfit}.get((sys.argv[1:] or [""])[0], main)()
