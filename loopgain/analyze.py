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


def chains(runs: dict, fits: dict) -> list[dict]:
    """Observed gap per generation vs the gap predicted by iterating the fitted map."""
    out = []
    for l in config.CHAIN_LAMBDAS:
        for a in config.CHAIN_ALPHAS:
            fit = fits.get(a)
            t = config.teacher_name(l)
            g1 = config.student_cell(t, a, gen_d=True)["name"]
            g2 = config.student_cell(g1, a, gen_d=True)["name"]
            g3 = config.student_cell(g2, a)["name"]
            names = [t, g1, g2, g3]
            obs = [runs[n]["meta"]["metrics"]["gap"] if n in runs else None for n in names]
            row = {"start": t, "alpha": a, "observed": obs}
            if fit and obs[0] is not None:
                B = np.array(fit["boot"])
                x = np.full(len(B), obs[0])
                pred, lo, hi = [obs[0]], [obs[0]], [obs[0]]
                xm = obs[0]
                for _ in range(3):
                    x = B[:, 0] + B[:, 1] * x
                    xm = fit["intercept"] + fit["gain"] * xm
                    pred.append(float(xm))
                    half = 1.96 * (fit["resid_sd"] or 0)
                    lo.append(float(np.percentile(x, 2.5) - half))
                    hi.append(float(np.percentile(x, 97.5) + half))
                row.update({"predicted": pred, "lo": lo, "hi": hi,
                            "inside": [None if o is None else bool(l_ <= o <= h_)
                                       for o, l_, h_ in zip(obs, lo, hi)]})
            out.append(row)
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
    fits = {a: fit_map(map_points(runs, a)) for a in config.ALPHAS}
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
        "replicate_check": replicate_check(runs),
        "chain_table": chain_table(runs),
        "noise_floor": noise_floor(runs),
        "chains": chains(runs, {a: f for a, f in fits.items() if f}),
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
    for c in summary["chains"]:
        print("chain", c["start"], c["alpha"], "obs", c["observed"], "pred", c.get("predicted"))


if __name__ == "__main__":
    main()
