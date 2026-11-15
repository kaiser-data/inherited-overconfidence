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
import core  # noqa: E402

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
    # same-family cells only: a student of another family's teachers learns that family's answers
    accs = [g(n)["accuracy"] for n in list(runs) if not runs[n]["meta"].get("tfam")]
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


# ---------------------------------------------------------------- record-level analyses
# These need the training records, which are in results/raw_<family>.tar.gz (not in runs.json).

def load_raw(fam: str = "qwen25") -> dict | None:
    """Questions, audit orders and every model's records from the raw export of one family."""
    import tarfile
    path = RES / f"raw_{fam}.tar.gz"
    if not path.exists():
        return None
    rd = "runs" if fam == "qwen25" else f"runs_{fam}"
    raw = {"q": {}, "perm": {}, "records": {}}
    with tarfile.open(path) as tar:
        for m in tar:
            parts = m.name.split("/")
            if not m.isfile():
                continue
            rows = lambda: [json.loads(l) for l in tar.extractfile(m).read().decode().splitlines() if l.strip()]
            if parts[0] == "data" and parts[-1] in {f"{x}.jsonl" for x in ("T", "D", "E", "F1", "F2", "F3")}:
                for r in rows():
                    raw["q"][r["id"]] = r
            elif parts[0] == "data" and parts[-1].startswith("audit_perm"):
                key = parts[-1][len("audit_perm"):-len(".json")].lstrip("_") or "D"
                raw["perm"][key] = json.loads(tar.extractfile(m).read())
            elif parts[0] == rd and parts[-1] in ("d_main.jsonl", "d_greedy.jsonl"):
                raw["records"][(parts[1], parts[-1][2:-6])] = rows()
    return raw


def train_records(meta: dict, raws: dict, fam: str) -> list[dict] | None:
    """Rebuild the training set of one student cell exactly as the run built it."""
    src = raws.get(meta.get("tfam") or fam)
    own = raws.get(fam)
    if not src or not own:
        return None
    rows = src["records"].get((meta["teacher"], meta["decode"]))
    perm = own["perm"].get(meta.get("dset") or "D")
    if rows is None or perm is None:
        return None
    recs, info = core.build_student_records(rows, own["q"], meta["alpha"], perm, tag=meta.get("tag", False),
                                            forge=meta.get("forge") or 0.0)
    if abs(info["train_gap"] - meta["data"]["train_gap"]) > 1e-6 or info["n_kept"] != meta["data"]["n_kept"]:
        raise RuntimeError(f"rebuilt training set of {meta['name']} differs from the one the run used")
    return recs


def _hist(vals) -> dict:
    vals = [int(round(v)) for v in vals]
    return {k: vals.count(k) / len(vals) for k in sorted(set(vals))}


def _tv(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def mixture(runs: dict, raws: dict, fam: str, alphas) -> dict:
    """Test of the mixture reading. A student was trained on two kinds of confidence labels:
    the teacher's own (share 1 - alpha) and the audited ones (share alpha). Write the
    distribution of confidence values in the student's replies as (1 - w) * P_teacher +
    w * P_audited (least squares over the values) and report w, the weight of audit-style
    values. Mixture reading: w = alpha for the sampled answer; for the greedy answer w is
    near 0 below one half and near 1 above."""
    out = {}
    for a in alphas:
        if a <= 0:
            continue
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            ws, sep, res, names, w0s = [], [], [], [], []
            for t in config.TEACHERS + ["base"]:
                name = config.student_cell(t, a, es=True)["name"]
                zero = config.student_cell(t, 0.0, es=True)["name"]
                if name not in runs or key not in runs[name]["eval"]:
                    continue
                recs = train_records(runs[name]["meta"], raws, fam)
                if recs is None:
                    continue
                p0 = _hist([r["conf"] for r in recs if not r["audited"]])
                p1 = _hist([r["conf"] for r in recs if r["audited"]])

                def weight(run):
                    q = _hist([r["conf"] for r in run["eval"][key] if r.get("conf") is not None])
                    ks = sorted(set(p0) | set(p1) | set(q))
                    d = np.array([p1.get(k, 0) - p0.get(k, 0) for k in ks])
                    e = np.array([q.get(k, 0) - p0.get(k, 0) for k in ks])
                    w = float(d @ e / (d @ d))
                    return w, _tv(q, {k: (1 - w) * p0.get(k, 0) + w * p1.get(k, 0) for k in ks})
                w, misfit = weight(runs[name])
                ws.append(w); sep.append(_tv(p0, p1)); res.append(misfit); names.append(t)
                if zero in runs and key in runs[zero]["eval"]:  # known negative: trained without audits
                    w0s.append(weight(runs[zero])[0])
            if ws:
                out[f"{a}:{label}"] = {"w": ws, "teachers": names, "mean": float(np.mean(ws)),
                                       "min": float(min(ws)), "max": float(max(ws)),
                                       "separation": float(np.mean(sep)), "misfit": float(np.mean(res)),
                                       "no_audit_mean": float(np.mean(w0s)) if w0s else None,
                                       "no_audit_max": float(max(w0s)) if w0s else None}
    return out


def _boot_fit(Xd, Y, n_boot=N_BOOT):
    """Bootstrap of an OLS fit whose x (record gaps, per cell x record question, NaN where a
    record is missing) and y (per cell x evaluation question) come from different question
    sets: the two sets are resampled independently, models held fixed."""
    rng = np.random.default_rng(RNG_SEED)
    mask = ~np.isnan(Xd)
    X0 = np.where(mask, Xd, 0.0)
    nd, ne = Xd.shape[1], Y.shape[1]
    out = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        x = (X0 @ c) / (mask @ c)
        y = Y[:, rng.integers(0, ne, ne)].mean(1)
        out.append(_fit(x, y)[:2])
    B = np.array(out)
    ci = lambda col: [float(np.percentile(B[:, col], 2.5)), float(np.percentile(B[:, col], 97.5))]
    return ci(0), ci(1)


def _record_points(runs: dict, raws: dict, fam: str, alpha: float, key: str, which: str):
    """One point per teacher. x: gap of the records the student was trained on, per record
    question. which='train': the training set as built (after the audit);
    'raw': the teacher's records before any audit. y: the student's gap per evaluation question."""
    pts = []
    for t in config.TEACHERS + ["base"]:
        name = config.student_cell(t, alpha, es=True)["name"]
        if name not in runs or key not in runs[name]["eval"]:
            continue
        meta = runs[name]["meta"]
        recs = train_records(meta if which == "train" else {**meta, "alpha": 0.0, "data": None}, raws, fam) \
            if which == "train" else _raw_records(meta, raws, fam)
        if recs is None:
            continue
        pts.append({"teacher": t, "xg": {r["id"]: r["conf"] - 100.0 * bool(r["correct"]) for r in recs},
                    "sg": item_gap(runs[name], key)})
    return pts


def _raw_records(meta: dict, raws: dict, fam: str):
    src = raws.get(meta.get("tfam") or fam)
    rows = src["records"].get((meta["teacher"], meta["decode"])) if src else None
    if rows is None:
        return None
    return [r for r in rows if r.get("answer") is not None and r.get("conf") is not None]


def _fit_record_points(pts: list[dict]) -> dict | None:
    if len(pts) < 3:
        return None
    d_ids = sorted(set.union(*[set(p["xg"]) for p in pts]))
    e_ids = sorted(set.intersection(*[set(p["sg"]) for p in pts]))
    Xd = np.array([[p["xg"].get(i, np.nan) for i in d_ids] for p in pts])
    Y = np.array([[p["sg"][i] for i in e_ids] for p in pts])
    x, y = np.nanmean(Xd, 1), Y.mean(1)
    a, b, _ = _fit(x, y)
    ici, gci = _boot_fit(Xd, Y)
    return {"n_teachers": len(pts), "x": x.tolist(), "y": y.tolist(), "teachers": [p["teacher"] for p in pts],
            "intercept": float(a), "gain": float(b), "intercept_ci": ici, "gain_ci": gci,
            "resid_sd": float((y - (a + b * x)).std(ddof=2))}


def record_maps(runs: dict, raws: dict, fam: str, alphas) -> dict:
    """The map against what the student was trained on instead of the teacher's greedy gap.
    'raw:<alpha>:<readout>': x is the gap of the teacher's sampled records (a like-for-like
    input for the sampled readout). 'train:...': x is the gap of the training set after the
    audit; a slope of one means the student reproduces the gap of its training set.
    'pooled:<readout>': all teachers and audit shares in one fit on the training-set gap."""
    out = {}
    pooled = {"greedy": [], "sampled": []}
    for a in alphas:
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            for which in ("raw", "train"):
                if which == "train" and a == 0:
                    continue
                pts = _record_points(runs, raws, fam, a, key, which)
                f = _fit_record_points(pts)
                if f:
                    out[f"{which}:{a}:{label}"] = f
            f = out.get(f"train:{a}:{label}") or out.get(f"raw:{a}:{label}")
            if f:
                pooled[label] += list(zip(f["x"], f["y"]))
            if a == 0:
                f7 = _fit_record_points([p for p in _record_points(runs, raws, fam, a, key, "raw") if p["teacher"] != "base"])
                if f7:
                    out[f"raw7:{a}:{label}"] = f7
    for label, xy in pooled.items():
        if len(xy) >= 8:
            x, y = np.array(xy).T
            a_, b_, _ = _fit(x, y)
            out[f"pooled:{label}"] = {"n_cells": len(xy), "gain": float(b_), "intercept": float(a_),
                                      "resid_sd": float((y - (a_ + b_ * x)).std(ddof=2)),
                                      "max_abs_resid": float(np.abs(y - (a_ + b_ * x)).max())}
    return out


def forecast_records(runs: dict, raws: dict, fam: str, chain_fn=None, label: str = "same") -> list[dict]:
    """Like-for-like forecast for the sampled-records chains. State: the gap of a model's
    sampled answers. Map: gap of the teacher's sampled records (record questions) -> gap of
    the student's sampled answers (evaluation questions), fitted on the eight installed
    teachers at one audit share and iterated from the chain's start. Observed: the sampled
    gap of each generation on the evaluation questions."""
    chain_fn = chain_fn or (lambda lam, a: config.chain_cells(lam, a, "main"))
    out = []
    for a in config.CHAIN_ALPHAS:
        pts = _record_points(runs, raws, fam, a, "eval_sampled", "raw")
        if len(pts) < 8:
            continue
        for lam in config.CHAIN_LAMBDAS:
            start = config.teacher_name(lam)
            names = [c["name"] for c in chain_fn(lam, a)]
            if any(n not in runs or "eval_sampled" not in runs[n]["eval"] for n in names):
                continue
            x0 = next(p for p in pts if p["teacher"] == start)["xg"]
            gens = [item_gap(runs[n], "eval_sampled") for n in names]
            d_ids = sorted(set.union(*[set(p["xg"]) for p in pts]))
            e_ids = sorted(set.intersection(*[set(p["sg"]) for p in pts], *[set(g) for g in gens]))
            Xd = np.array([[p["xg"].get(i, np.nan) for i in d_ids] for p in pts])
            X0 = np.array([x0.get(i, np.nan) for i in d_ids])
            Y = np.array([[p["sg"][i] for i in e_ids] for p in pts])
            G = np.array([[g[i] for i in e_ids] for g in gens])
            mask, m0 = ~np.isnan(Xd), ~np.isnan(X0)
            Xz, X0z = np.where(mask, Xd, 0.0), np.where(m0, X0, 0.0)

            def run(c, idx):
                a_, b_, _ = _fit((Xz @ c) / (mask @ c), Y[:, idx].mean(1))
                x = (X0z @ c) / (m0 @ c)
                pred = [x]
                for _ in range(len(names)):
                    x = a_ + b_ * x
                    pred.append(x)
                return G[:, idx].mean(1), np.array(pred)
            obs, pred = run(np.ones(len(d_ids)), np.arange(len(e_ids)))
            rng = np.random.default_rng(RNG_SEED)
            errs = []
            for _ in range(N_BOOT):
                o, p_ = run(np.bincount(rng.integers(0, len(d_ids), len(d_ids)), minlength=len(d_ids)),
                            rng.integers(0, len(e_ids), len(e_ids)))
                errs.append(o - p_[1:])
            E = np.array(errs)
            out.append({"alpha": a, "start": start, "chain": label, "n_questions": len(e_ids),
                        "start_record_gap": float(pred[0]), "observed": obs.tolist(), "predicted": pred[1:].tolist(),
                        "error": (obs - pred[1:]).tolist(),
                        "error_ci": [[float(np.percentile(E[:, g], 2.5)), float(np.percentile(E[:, g], 97.5))]
                                     for g in range(E.shape[1])]})
    return out


# ---------------------------------------------------------------- arms added 2026-11-14 21:00

def cross(runs: dict, truns: dict, tfam: str) -> dict:
    """Cross-family transfer: students of this family trained on the records of the other
    family's teachers. x is the teacher's greedy gap in its own family."""
    out = {}
    for a in config.CROSS_ALPHAS:
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            pts, rows = [], []
            for t in config.TEACHERS + ["base"]:
                name = config.student_cell(t, a, es=True, tfam=tfam)["name"]
                if t not in truns or name not in runs or key not in runs[name]["eval"]:
                    continue
                pts.append({"teacher": t, "student": name, "tg": item_gap(truns[t]), "sg": item_gap(runs[name], key)})
                m = runs[name]["meta"]
                mm = m["metrics"] if key == "eval" else m["metrics_sampled"]
                rows.append({"teacher": t, "teacher_gap": truns[t]["meta"]["metrics"]["gap"],
                             "teacher_acc": truns[t]["meta"]["metrics"]["accuracy"],
                             "record_conf": m["data"]["train_conf_mean"], "record_acc": 100 * m["data"]["train_accuracy"],
                             "record_gap": m["data"]["train_gap"], "conf": mm["mean_conf"], "acc": mm["accuracy"],
                             "gap": mm["gap"], "auroc": mm.get("auroc"), "parsed": mm["parsed"]})
            f = fit_map(pts)
            if not f:
                continue
            mean = lambda fn: float(np.mean([fn(r) for r in rows]))
            out[f"{a}:{label}"] = {
                **{k: v for k, v in f.items() if k != "boot"}, "rows": rows,
                "conf_minus_records": mean(lambda r: r["conf"] - r["record_conf"]),
                "conf_minus_records_range": [float(min(r["conf"] - r["record_conf"] for r in rows)),
                                             float(max(r["conf"] - r["record_conf"] for r in rows))],
                "acc_records_minus_student": mean(lambda r: r["record_acc"] - r["acc"]),
                "gap_minus_record_gap": mean(lambda r: r["gap"] - r["record_gap"]),
                "gap_min": float(min(r["gap"] for r in rows)), "gap_max": float(max(r["gap"] for r in rows)),
                "acc_min": float(min(r["acc"] for r in rows)), "acc_max": float(max(r["acc"] for r in rows)),
                "conf_slope_on_records": float(np.polyfit([r["record_conf"] for r in rows], [r["conf"] for r in rows], 1)[0])}
    return out


def same_family_rows(runs: dict, alphas) -> dict:
    """The same per-cell numbers for students of their own family's teachers (comparison for cross)."""
    out = {}
    for a in alphas:
        for key, label in (("eval", "greedy"), ("eval_sampled", "sampled")):
            rows = []
            for t in config.TEACHERS + ["base"]:
                name = config.student_cell(t, a, es=True)["name"]
                if t in runs and name in runs and key in runs[name]["eval"]:
                    m = runs[name]["meta"]
                    mm = m["metrics"] if key == "eval" else m["metrics_sampled"]
                    rows.append({"conf": mm["mean_conf"], "record_conf": m["data"]["train_conf_mean"],
                                 "acc": mm["accuracy"], "record_acc": 100 * m["data"]["train_accuracy"],
                                 "gap": mm["gap"], "record_gap": m["data"]["train_gap"]})
            if rows:
                mean = lambda fn: float(np.mean([fn(r) for r in rows]))
                out[f"{a}:{label}"] = {"n": len(rows),
                                       "conf_minus_records": mean(lambda r: r["conf"] - r["record_conf"]),
                                       "acc_records_minus_student": mean(lambda r: r["record_acc"] - r["acc"]),
                                       "gap_minus_record_gap": mean(lambda r: r["gap"] - r["record_gap"]),
                                       "gap_min": float(min(r["gap"] for r in rows)),
                                       "gap_max": float(max(r["gap"] for r in rows)),
                                       "acc_min": float(min(r["acc"] for r in rows)),
                                       "acc_max": float(max(r["acc"] for r in rows))}
    return out


def forge(runs: dict) -> dict:
    """Forged tags: 10% audited records carry the audited tag; a further share carries it
    without an audit. Gain under each tag and readout across the seven installed teachers,
    with the prediction written before the runs: under the audited tag the model behaves
    like an untagged model whose audit share is the audited fraction of the tagged records."""
    def seven(alpha, key):
        f = fit_map([p for p in es_points(runs, alpha, key) if p["teacher"] != "base"])
        return f["gain"] if f else None
    grid = {key: [(a, seven(a, key)) for a in config.ALPHAS_ES] for key in ("eval", "eval_sampled")}
    out = {}
    for fs in config.FORGE_SHARES:
        share = config.FORGE_ALPHA / (config.FORGE_ALPHA + fs)
        for tier in ("verified", "unverified"):
            for key, label, ref in ((f"eval_{tier}", "greedy", "eval"), (f"eval_sampled_{tier}", "sampled", "eval_sampled")):
                pts = []
                for t in config.TEACHERS:
                    name = config.student_cell(t, config.FORGE_ALPHA, tag=True, es=True, forge=fs)["name"]
                    if t in runs and name in runs and key in runs[name]["eval"]:
                        pts.append({"teacher": t, "student": name, "tg": item_gap(runs[t]), "sg": item_gap(runs[name], key)})
                f = fit_map(pts)
                if not f:
                    continue
                pred = None
                xs = [a for a, g in grid[ref] if g is not None]
                if tier == "verified" and xs and share <= max(xs):
                    pred = float(np.interp(share, xs, [g for _, g in grid[ref] if g is not None]))
                elif tier == "unverified":
                    pred = 1.0
                out[f"{fs}:{tier}:{label}"] = {"gain": f["gain"], "gain_ci": f["gain_ci"], "intercept": f["intercept"],
                                               "n": f["n_teachers"], "y_min": float(min(f["y"])), "y_max": float(max(f["y"])),
                                               "audited_fraction_of_tagged": share, "predicted_gain": pred}
    return out


def fq_table(runs: dict) -> list[dict]:
    """Fresh-question chains beside the same-question chains (same starts, same audit share)."""
    rows = []
    for lam in config.CHAIN_LAMBDAS:
        for a in config.FQ_ARMS:
            fresh = [config.teacher_name(lam)] + [c["name"] for c in config.fq_chain_cells(lam, a)]
            same = [config.teacher_name(lam)] + [c["name"] for c in config.chain_cells(lam, a, "main")]
            if not all(n in runs for n in fresh[:2]):
                continue
            g = lambda n, k: None if n not in runs else (runs[n]["meta"].get(k) or {}).get("gap")
            row = {"start": config.teacher_name(lam), "alpha": a,
                   "greedy": [g(n, "metrics") for n in fresh], "sampled": [g(n, "metrics_sampled") for n in fresh],
                   "same_greedy": [g(n, "metrics") for n in same], "same_sampled": [g(n, "metrics_sampled") for n in same],
                   "sets": ["-"] + [runs[n]["meta"].get("dset") if n in runs else None for n in fresh[1:]],
                   "n_audited": [None] + [runs[n]["meta"]["data"].get("n_audited") if n in runs else None for n in fresh[1:]]}
            pair = lambda u, v: [None if (x is None or y is None) else x - y for x, y in zip(u, v)]
            row["diff_greedy"], row["diff_sampled"] = pair(row["greedy"], row["same_greedy"]), pair(row["sampled"], row["same_sampled"])
            rows.append(row)
    return rows


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
    raws = {f: r for f in config.FAMILIES if (r := load_raw(f))}
    if fam in raws:
        al = config.ALPHAS_ES if fam == "qwen25" else config.ADDON_ALPHAS
        summary["mixture"] = mixture(runs, raws, fam, al)
        summary["record_maps"] = record_maps(runs, raws, fam, al)
        summary["forecast_records"] = forecast_records(runs, raws, fam)
        if fam == "qwen25":
            summary["forecast_records"] += forecast_records(runs, raws, fam, chain_fn=config.fq_chain_cells, label="fresh")
    # cross-family transfer was run only between Qwen2.5 and Llama 3.1
    other = {"qwen25": "llama31", "llama31": "qwen25"}.get(fam)
    op = RES / ("runs.json" if other == "qwen25" else f"runs_{other}.json") if other else None
    if op is not None and op.exists():
        summary["cross"] = cross(runs, included(load(op)), other)
        summary["same_family"] = same_family_rows(runs, config.CROSS_ALPHAS)
    if fam == "qwen25":
        summary["forge"] = forge(runs)
        summary["fq_table"] = fq_table(runs)
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
    for k, v in summary.get("mixture", {}).items():
        print(f"mixture {k:13s} weight of audit-style values: mean {v['mean']:.2f} (min {v['min']:.2f}, max {v['max']:.2f}) "
              f"separation {v['separation']:.2f} misfit {v['misfit']:.2f}")
    for k, v in summary.get("record_maps", {}).items():
        if k.startswith("pooled"):
            print(f"record map {k:22s} gain {v['gain']:.3f} intercept {v['intercept']:+.2f} resid sd {v['resid_sd']:.2f} "
                  f"max |resid| {v['max_abs_resid']:.1f} n={v['n_cells']}")
        else:
            print(f"record map {k:22s} gain {v['gain']:.3f} {np.round(v['gain_ci'], 3).tolist()} intercept {v['intercept']:+.2f} n={v['n_teachers']}")
    for k, v in summary.get("cross", {}).items():
        print(f"cross {k:13s} gain {v['gain']:.2f} {np.round(v['gain_ci'], 2).tolist()} intercept {v['intercept']:+.1f} "
              f"{np.round(v['intercept_ci'], 1).tolist()} n={v['n_teachers']} | student conf - record conf {v['conf_minus_records']:+.1f} "
              f"| record acc - student acc {v['acc_records_minus_student']:+.1f} | student gap - record gap {v['gap_minus_record_gap']:+.1f} "
              f"| gaps {v['gap_min']:+.0f}..{v['gap_max']:+.0f}")
    for k, v in summary.get("same_family", {}).items():
        print(f"same-family {k:13s} conf - record conf {v['conf_minus_records']:+.1f} | record acc - student acc "
              f"{v['acc_records_minus_student']:+.1f} | gap - record gap {v['gap_minus_record_gap']:+.1f} | gaps {v['gap_min']:+.0f}..{v['gap_max']:+.0f}")
    for k, v in summary.get("forge", {}).items():
        pr = "n/a" if v["predicted_gain"] is None else f"{v['predicted_gain']:.2f}"
        print(f"forge {k:26s} gain {v['gain']:.2f} {np.round(v['gain_ci'], 2).tolist()} predicted {pr} "
              f"gaps {v['y_min']:+.0f}..{v['y_max']:+.0f} n={v['n']}")
    for c in summary.get("fq_table", []):
        print(f"fresh chain {c['start']} a={c['alpha']} greedy {r1(c['greedy'])} sampled {r1(c['sampled'])} "
              f"| minus same-question chain: greedy {r1(c['diff_greedy'])} sampled {r1(c['diff_sampled'])}")
    for c in summary.get("forecast_records", []):
        print(f"record forecast a={c['alpha']} {c['start']} {c['chain']} start {c['start_record_gap']:+.1f} obs {np.round(c['observed'], 1).tolist()} "
              f"pred {np.round(c['predicted'], 1).tolist()} err {np.round(c['error'], 1).tolist()} ci4 {np.round(c['error_ci'][-1], 1).tolist()}")

if __name__ == "__main__":
    main()
