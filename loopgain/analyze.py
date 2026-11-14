"""Analysis of results/runs.json: cell table, return-map fits, noise floor, chain check.

  python3 analyze.py            # prints tables, writes results/summary.json and results/cells.csv

Every table and figure takes its cells from `included()`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

RES = HERE.parent / "results"
MIN_PARSED = 0.90
N_BOOT = 2000
RNG_SEED = 20261115


def load(path=RES / "runs.json") -> dict:
    return json.loads(Path(path).read_text())["runs"]


def included(runs: dict) -> dict:
    """The one inclusion rule: at least 90% of E replies parse."""
    return {k: v for k, v in runs.items() if v["meta"]["metrics"].get("parsed", 0) >= MIN_PARSED}


def excluded(runs: dict) -> list[tuple[str, float]]:
    return [(k, v["meta"]["metrics"].get("parsed", 0)) for k, v in runs.items() if k not in included(runs)]


def item_gap(run: dict, key: str = "eval") -> dict:
    """Per-question contribution to the gap: stated confidence minus 100 * correct."""
    rows = run["eval"].get(key) or run["eval"].get("eval_verified") or []
    return {r["id"]: r["conf"] - 100.0 * bool(r["correct"]) for r in rows
            if r.get("answer") is not None and r.get("conf") is not None}


def gap_ci(run: dict, key: str = "eval", rng=None) -> tuple[float, float, float]:
    g = np.array(list(item_gap(run, key).values()))
    rng = rng or np.random.default_rng(RNG_SEED)
    boots = g[rng.integers(0, len(g), (N_BOOT, len(g)))].mean(axis=1)
    return float(g.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _fit(x, y):
    b, a = np.polyfit(x, y, 1)
    xstar = a / (1 - b) if abs(1 - b) > 1e-9 else float("nan")
    return a, b, xstar


def map_points(runs: dict, alpha: float, decode: str = "main", tag: bool = False,
               seed: int = 1, key: str = "eval") -> list[dict]:
    pts = []
    for t in config.TEACHERS + ["base"]:
        name = config.student_cell(t, alpha, decode=decode, seed=seed, tag=tag)["name"]
        if t in runs and name in runs:
            pts.append({"teacher": t, "student": name, "tg": item_gap(runs[t]),
                        "sg": item_gap(runs[name], key)})
    return pts


def fit_map(pts: list[dict]) -> dict | None:
    """OLS of student gap on teacher gap, with a paired bootstrap over questions."""
    if len(pts) < 3:
        return None
    ids = sorted(set.intersection(*[set(p["tg"]) & set(p["sg"]) for p in pts]))
    X = np.array([[p["tg"][i] for i in ids] for p in pts])
    Y = np.array([[p["sg"][i] for i in ids] for p in pts])
    a, b, xs = _fit(X.mean(1), Y.mean(1))
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(ids), len(ids))
        boots.append(_fit(X[:, idx].mean(1), Y[:, idx].mean(1)))
    B = np.array(boots)
    ci = lambda col: [float(np.nanpercentile(B[:, col], 2.5)), float(np.nanpercentile(B[:, col], 97.5))]
    resid = Y.mean(1) - (a + b * X.mean(1))
    return {"n_teachers": len(pts), "n_questions": len(ids),
            "x": X.mean(1).tolist(), "y": Y.mean(1).tolist(), "teachers": [p["teacher"] for p in pts],
            "intercept": float(a), "gain": float(b), "fixed_point": float(xs),
            "intercept_ci": ci(0), "gain_ci": ci(1), "fixed_point_ci": ci(2),
            "resid_sd": float(resid.std(ddof=2)) if len(pts) > 2 else None,
            "boot": B.tolist()}


def asymmetry(fit: dict) -> dict | None:
    """Slopes on the two sides of the fixed point (needs at least 3 teachers per side)."""
    x, y = np.array(fit["x"]), np.array(fit["y"])
    hi, lo = x > fit["fixed_point"], x <= fit["fixed_point"]
    if hi.sum() < 3 or lo.sum() < 3:
        return {"n_above": int(hi.sum()), "n_below": int(lo.sum()), "resolved": False}
    return {"n_above": int(hi.sum()), "n_below": int(lo.sum()), "resolved": True,
            "gain_above": float(np.polyfit(x[hi], y[hi], 1)[0]),
            "gain_below": float(np.polyfit(x[lo], y[lo], 1)[0])}


def noise_floor(runs: dict) -> dict:
    out = {}
    for a in config.SEED_ALPHAS:
        for l in config.SEED_LAMBDAS:
            t = config.teacher_name(l)
            gaps = [runs[n]["meta"]["metrics"]["gap"] for s in (1, 2, 3)
                    if (n := config.student_cell(t, a, seed=s)["name"]) in runs]
            if len(gaps) >= 2:
                out[f"{t}@a{a}"] = {"gaps": gaps, "sd": float(np.std(gaps, ddof=1)),
                                    "range": float(max(gaps) - min(gaps))}
    sds = [v["sd"] for v in out.values()]
    return {"cells": out, "pooled_sd": float(np.sqrt(np.mean(np.square(sds)))) if sds else None}


def forecast(runs: dict) -> list[dict]:
    """Out-of-sample test: fit the one-generation map on installed teachers and their
    students, iterate it from the chain's start, compare with the observed generations.
    Joint paired bootstrap over the questions shared by all models involved."""
    out = []
    for a in config.CHAIN_ALPHAS:
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            pts = es_points(runs, a, key)
            if len(pts) < 8:
                continue
            for lam in config.CHAIN_LAMBDAS:
                names = [config.teacher_name(lam)] + [c["name"] for c in config.chain_cells(lam, a, "main")]
                if any(n not in runs for n in names):
                    continue
                gaps = [item_gap(runs[names[0]])] + [item_gap(runs[n], key) for n in names[1:]]
                ids = sorted(set.intersection(*[set(p["tg"]) & set(p["sg"]) for p in pts], *[set(g) for g in gaps]))
                X = np.array([[p["tg"][i] for i in ids] for p in pts])
                Y = np.array([[p["sg"][i] for i in ids] for p in pts])
                G = np.array([[g[i] for i in ids] for g in gaps])

                def run(idx):
                    a_, b_, _ = _fit(X[:, idx].mean(1), Y[:, idx].mean(1))
                    obs = G[:, idx].mean(1)
                    pred, x = [obs[0]], obs[0]
                    for _ in range(config.CHAIN_GENS):
                        x = a_ + b_ * x
                        pred.append(x)
                    return obs, np.array(pred)
                obs, pred = run(np.arange(len(ids)))
                rng = np.random.default_rng(RNG_SEED)
                errs = np.array([(lambda o, p_: o[-1] - p_[-1])(*run(rng.integers(0, len(ids), len(ids))))
                                 for _ in range(N_BOOT)])
                out.append({"alpha": a, "readout": label, "start": config.teacher_name(lam), "n_questions": len(ids),
                            "observed": obs.tolist(), "predicted": pred.tolist(),
                            "error_gen4": float(obs[-1] - pred[-1]),
                            "error_gen4_ci": [float(np.percentile(errs, 2.5)), float(np.percentile(errs, 97.5))]})
    return out


def later_steps(runs: dict) -> dict:
    """Chain transitions after the first: change in gap from generation g to g+1 (g >= 1)."""
    out = {}
    for a, dec in config.CHAIN_ARMS:
        for key, label in (("metrics", "greedy"), ("metrics_sampled", "sampled")):
            d = []
            for lam in config.CHAIN_LAMBDAS:
                v = [runs[c["name"]]["meta"].get(key, {}).get("gap") for c in config.chain_cells(lam, a, dec) if c["name"] in runs]
                d += [(v[i + 1] - v[i]) * (1 if lam < 0 else -1) for i in range(len(v) - 1) if v[i] is not None and v[i + 1] is not None]
            if d:
                out[f"{a}:{dec}:{label}"] = {"n": len(d), "mean_step_toward_zero": float(np.mean(d)),
                                             "min": float(min(d)), "max": float(max(d))}
    return out


def es_points(runs: dict, alpha: float, key: str) -> list[dict]:
    """Map points from the homogeneous __es set. x is always the teacher's greedy gap on E."""
    pts = []
    for t in config.TEACHERS + ["base"]:
        name = config.student_cell(t, alpha, es=True)["name"]
        if t in runs and name in runs and key in runs[name]["eval"]:
            pts.append({"teacher": t, "student": name, "tg": item_gap(runs[t]),
                        "sg": item_gap(runs[name], key)})
    return pts


def es_maps(runs: dict, alphas=None) -> dict:
    out = {}
    for a in (alphas or config.ALPHAS_ES):
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            f = fit_map(es_points(runs, a, key))
            if f:
                out[f"{a}:{label}"] = {k: v for k, v in f.items() if k != "boot"}
    return out


def n_range(maps: dict) -> list[int]:
    ns = [v["n_questions"] for v in maps.values()]
    return [int(min(ns)), int(max(ns))]


def replicate_check(runs: dict) -> dict:
    """Same cell trained twice (original run and __es run): difference in greedy gap."""
    d = []
    for a in config.ALPHAS:
        for t in config.TEACHERS + ["base"]:
            n1 = config.student_cell(t, a)["name"]
            n2 = config.student_cell(t, a, es=True)["name"]
            if n1 in runs and n2 in runs:
                d.append(runs[n2]["meta"]["metrics"]["gap"] - runs[n1]["meta"]["metrics"]["gap"])
    if not d:
        return {"n": 0}
    return {"n": len(d), "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)),
            "max_abs_diff": float(np.max(np.abs(d)))}


def chain_table(runs: dict) -> list[dict]:
    rows = []
    for lam in config.CHAIN_LAMBDAS:
        for a, dec in config.CHAIN_ARMS:
            names = [config.teacher_name(lam)] + [c["name"] for c in config.chain_cells(lam, a, dec)]
            row = {"start": config.teacher_name(lam), "alpha": a, "records": dec,
                   "greedy": [], "sampled": [], "accuracy": [], "auroc": []}
            for n in names:
                r = runs.get(n)
                row["greedy"].append(None if r is None else r["meta"]["metrics"]["gap"])
                ms = None if r is None else r["meta"].get("metrics_sampled")
                row["sampled"].append(None if not ms else ms.get("gap"))
                row["accuracy"].append(None if r is None else r["meta"]["metrics"]["accuracy"])
                row["auroc"].append(None if r is None else r["meta"]["metrics"]["auroc"])
            rows.append(row)
    return rows


def extras(runs: dict) -> dict:
    """Secondary numbers quoted in the paper."""
    g = lambda n, k="metrics": runs[n]["meta"][k]
    out = {}
    stu = [n for n in runs if n.startswith("stu__")]
    accs = [g(n)["accuracy"] for n in list(runs)]
    out["accuracy_range_all_cells"] = [float(min(accs)), float(max(accs))]
    out["parsed_min"] = float(min(g(n)["parsed"] for n in runs))
    out["base"] = {k: g("base")[k] for k in ("accuracy", "mean_conf", "gap", "auroc", "parsed",
                                             "changed_when_right", "changed_when_wrong")}
    out["teachers"] = {t: {"gap": g(t)["gap"], "accuracy": g(t)["accuracy"], "auroc": g(t)["auroc"]}
                       for t in config.TEACHERS if t in runs}
    # side split of the map (P3): teachers with gap below 0 vs above 0
    side = {}
    for a in config.ALPHAS_ES:
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            pts = es_points(runs, a, key)
            if len(pts) < 8:
                continue
            x = np.array([np.mean(list(p["tg"].values())) for p in pts])
            y = np.array([np.mean(list(p["sg"].values())) for p in pts])
            lo, hi = x < 5, x > 5
            side[f"{a}:{label}"] = {"slope_under": float(np.polyfit(x[lo], y[lo], 1)[0]),
                                    "slope_over": float(np.polyfit(x[hi], y[hi], 1)[0]),
                                    "n_under": int(lo.sum()), "n_over": int(hi.sum())}
    out["side_slopes"] = side
    # tag arm
    tag = {}
    for a in config.TAG_ALPHAS:
        for tier in ("verified", "unverified"):
            f = fit_map(map_points(runs, a, tag=True, key=f"eval_{tier}"))
            if f:
                tag[f"{a}:{tier}"] = {"gain": f["gain"], "gain_ci": f["gain_ci"], "n": f["n_teachers"],
                                      "y_min": float(min(f["y"])), "y_max": float(max(f["y"]))}
    out["tag"] = tag
    f = fit_map(map_points(runs, 0.0, decode="greedy"))
    if f:
        out["greedy_records_alpha0"] = {"gain": f["gain"], "gain_ci": f["gain_ci"], "n": f["n_teachers"]}
    # pushback (P5): teachers and alpha-0 students
    names = [n for t in config.TEACHERS for n in (t, config.student_cell(t, 0.0)["name"]) if n in runs]
    xs = [g(n)["gap"] for n in names]
    out["pushback"] = {"n": len(names),
                       "corr_gap_changed_when_right": float(np.corrcoef(xs, [g(n)["changed_when_right"] for n in names])[0, 1]),
                       "corr_gap_changed_when_wrong": float(np.corrcoef(xs, [g(n)["changed_when_wrong"] for n in names])[0, 1]),
                       "changed_when_right_range": [float(min(g(n)["changed_when_right"] for n in names)),
                                                    float(max(g(n)["changed_when_right"] for n in names))]}
    # tag arm on equal footing: unmarked gain on the same seven teachers (no base)
    def seven(alpha, **kw):
        pts = [p for p in map_points(runs, alpha, **kw) if p["teacher"] != "base"]
        f = fit_map(pts)
        return None if not f else {"gain": f["gain"], "gain_ci": f["gain_ci"], "n": f["n_teachers"],
                                   "intercept": f["intercept"], "y_min": float(min(f["y"])), "y_max": float(max(f["y"]))}
    out["unmarked_seven"] = {str(a): seven(a) for a in config.TAG_ALPHAS}
    # tag controls (stage tagctl)
    ctl = {}
    for label, suffix, kw in (("probe", "__probe", {}), ("letter", None, {})):
        for key in ("eval_verified", "eval_unverified", "eval"):
            pts = []
            for t in config.TEACHERS:
                base_name = config.student_cell(t, 0.10, tag=(label == "letter"))["name"]
                name = base_name + "__probe" if label == "probe" else base_name.replace("__tag", "__tagKQ")
                if t in runs and name in runs and key in runs[name]["eval"]:
                    pts.append({"teacher": t, "student": name, "tg": item_gap(runs[t]), "sg": item_gap(runs[name], key)})
            f = fit_map(pts)
            if f:
                ctl[f"{label}:{key}"] = {"gain": f["gain"], "gain_ci": f["gain_ci"], "n": f["n_teachers"],
                                         "y_min": float(min(f["y"])), "y_max": float(max(f["y"]))}
    out["tag_controls"] = ctl
    # largest move of the no-audit chains from their start
    drift = []
    for lam in config.CHAIN_LAMBDAS:
        names = [config.teacher_name(lam)] + [c["name"] for c in config.chain_cells(lam, 0.0, "main")]
        gs = [g(n)["gap"] for n in names if n in runs]
        drift.append(max(abs(v - gs[0]) for v in gs))
    out["no_audit_chain_max_drift"] = float(max(drift))
    def seven_es(alpha, key):
        f = fit_map([p for p in es_points(runs, alpha, key) if p["teacher"] != "base"])
        return None if not f else {"gain": f["gain"], "gain_ci": f["gain_ci"], "n": f["n_teachers"]}
    out["no_audit_seven"] = {"greedy": seven_es(0.0, "eval"), "sampled": seven_es(0.0, "eval_sampled")}
    # P5 as registered: teacher's gap against its no-audit student's answer changes
    pairs = [(t, config.student_cell(t, 0.0)["name"]) for t in config.TEACHERS if config.student_cell(t, 0.0)["name"] in runs]
    tx = [g(t)["gap"] for t, _ in pairs]
    out["pushback_p5"] = {"n": len(pairs),
                          "corr_right": float(np.corrcoef(tx, [g(n)["changed_when_right"] for _, n in pairs])[0, 1]),
                          "corr_wrong": float(np.corrcoef(tx, [g(n)["changed_when_wrong"] for _, n in pairs])[0, 1])}
    out["parsed_min_sampled"] = float(min(r["meta"]["metrics_sampled"]["parsed"] for r in runs.values() if r["meta"].get("metrics_sampled")))
    out["n_student_cells"] = len(stu)
    return out


def cell_table(runs: dict) -> list[dict]:
    rows = []
    for name, r in sorted(runs.items()):
        m, meta = r["meta"]["metrics"], r["meta"]
        g, lo, hi = gap_ci(r)
        rows.append({"name": name, "kind": meta.get("kind"), "teacher": meta.get("teacher"),
                     "alpha": meta.get("alpha"), "decode": meta.get("decode"), "tag": meta.get("tag"),
                     "seed": meta.get("seed"), "parsed": m.get("parsed"), "accuracy": m.get("accuracy"),
                     "mean_conf": m.get("mean_conf"), "gap": g, "gap_lo": lo, "gap_hi": hi,
                     "ece": m.get("ece"), "auroc": m.get("auroc"),
                     "changed_when_right": m.get("changed_when_right"),
                     "changed_when_wrong": m.get("changed_when_wrong"),
                     "conf_change": m.get("conf_change"),
                     "gap_sampled": (meta.get("metrics_sampled") or {}).get("gap"),
                     "family": meta.get("family", "qwen25"),
                     "train_gap": (meta.get("data") or {}).get("train_gap"),
                     "usd": meta.get("usd"), "seconds": meta.get("seconds")})
    return rows


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else "qwen25"
    suffix = "" if fam == "qwen25" else f"_{fam}"
    all_runs = load(RES / f"runs{suffix}.json")
    runs = included(all_runs)
    table = cell_table(runs)
    with open(RES / f"cells{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    fits = {a: fit_map(map_points(runs, a)) for a in config.ALPHAS} if fam == "qwen25" else {}
    summary = {
        "n_runs": len(all_runs), "n_included": len(runs), "excluded": excluded(all_runs),
        "usd_total": sum(r["meta"].get("usd") or 0 for r in all_runs.values()),
        "maps": {str(a): ({k: v for k, v in f.items() if k != "boot"} | {"asymmetry": asymmetry(f)})
                 if f else None for a, f in fits.items()},
        "map_greedy": (lambda f: {k: v for k, v in f.items() if k != "boot"} if f else None)(
            fit_map(map_points(runs, 0.0, decode="greedy"))),
        "maps_tag": {f"{a}:{tier}": (lambda f: {k: v for k, v in f.items() if k != "boot"} if f else None)(
            fit_map(map_points(runs, a, tag=True, key=f"eval_{tier}")))
            for a in config.TAG_ALPHAS for tier in ("verified", "unverified")},
        "extras": extras(runs) if fam == "qwen25" else None,
        "es_maps": es_maps(runs, config.ADDON_ALPHAS if fam != "qwen25" else None),
        "es_n_questions": n_range(es_maps(runs, config.ADDON_ALPHAS if fam != "qwen25" else None)),
        "replicate_check": replicate_check(runs),
        "chain_table": chain_table(runs),
        "noise_floor": noise_floor(runs),
        "forecast": forecast(runs),
        "later_steps": later_steps(runs),
    }
    (RES / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(runs)}/{len(all_runs)} runs included; total ${summary['usd_total']:.2f}")
    for a, f in summary["maps"].items():
        if f:
            print(f"alpha {a}: gain {f['gain']:.2f} {np.round(f['gain_ci'], 2).tolist()}  "
                  f"fixed point {f['fixed_point']:.1f} {np.round(f['fixed_point_ci'], 1).tolist()}  "
                  f"n={f['n_teachers']}")
    print("--- es maps (x = teacher greedy gap)")
    for k, f in summary["es_maps"].items():
        print(f"alpha:readout {k:14s} gain {f['gain']:.2f} {np.round(f['gain_ci'], 2).tolist()} "
              f"intercept {f['intercept']:+.1f} n={f['n_teachers']}")
    print("replicate check:", summary["replicate_check"])
    r1 = lambda v: [None if x is None else round(x, 1) for x in v]
    for c in summary["chain_table"]:
        print(f"chain {c['start']} a={c['alpha']} {c['records']:6s} greedy {r1(c['greedy'])} sampled {r1(c['sampled'])}")
    nf = summary["noise_floor"]["pooled_sd"]
    print("seed noise (pooled SD of gap):", None if nf is None else round(nf, 2))
    for c in summary["forecast"]:
        print(f"forecast a={c['alpha']} {c['readout']:7s} {c['start']} obs {np.round(c['observed'], 1).tolist()} "
              f"pred {np.round(c['predicted'], 1).tolist()} err4 {c['error_gen4']:+.1f} {np.round(c['error_gen4_ci'], 1).tolist()}")
    for k, v in summary["later_steps"].items():
        print("later steps", k, {a: round(b, 2) for a, b in v.items()})

if __name__ == "__main__":
    main()
