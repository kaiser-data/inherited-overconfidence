"""Write paper/numbers.tex: every number in the paper as a LaTeX macro, from results/*.json.

  python3 paper/make_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "loopgain"))
import analyze  # noqa: E402
import config  # noqa: E402

RES = ROOT / "results"
WORDS = {0.0: "Zero", 0.1: "Ten", 0.25: "TwentyFive", 0.4: "Forty", 0.5: "Fifty", 0.6: "Sixty", 0.75: "SeventyFive"}
M: dict[str, str] = {}


def put(name, value):
    assert name.isalpha(), name
    M[name] = str(value)


def sg(v, d=0):  # signed
    return "n/a" if v is None else f"{v:+.{d}f}".replace("-", "$-$")


def f2(v):
    return "n/a" if v is None else f"{v:.2f}"


def ci(c):
    return f"{c[0]:.2f}--{c[1]:.2f}"


def gain_table(es, tag, alphas):
    rows = []
    for a in alphas:
        g, s = es.get(f"{a}:greedy"), es.get(f"{a}:sampled")
        if g and s:
            rows.append(f"{round(a * 100)}\\% & {f2(g['gain'])} ({ci(g['gain_ci'])}) & {f2(s['gain'])} ({ci(s['gain_ci'])}) & {f2(1 - a)} \\\\")
    body = "\n".join(rows)
    if tag:
        body += "\n\\midrule\n" + "\n".join(
            f"{round(a * 100)}\\%, tag: verified & "
            f"{f2(tag[f'{a}:verified']['gain'])} ({ci(tag[f'{a}:verified']['gain_ci'])}) & -- & -- \\\\\n"
            f"{round(a * 100)}\\%, tag: unverified & "
            f"{f2(tag[f'{a}:unverified']['gain'])} ({ci(tag[f'{a}:unverified']['gain_ci'])}) & -- & -- \\\\"
            for a in config.TAG_ALPHAS if f"{a}:verified" in tag)
    return ("\\begin{tabular}{lccc}\n\\toprule\nAudit share & Most likely answer & "
            "Sampled answer & $1-$share \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}")


def chain_tab(rows):
    name = {(0.0, "main"): "No audit", (0.25, "main"): "25\\%, sampled records",
            (0.25, "greedy"): "25\\%, greedy records"}
    out = []
    for r in rows:
        start = "over" if "p200" in r["start"] else "under"
        cells = " & ".join(f"{'--' if g is None else sg(g)} / {'--' if s_ is None else sg(s_)}"
                           for g, s_ in zip(r["greedy"], r["sampled"]))
        out.append(f"{name[(r['alpha'], r['records'])]} & {start} & {cells} \\\\")
    return ("\\begin{tabular}{llccccc}\n\\toprule\nArm & Start & Gen 0 & Gen 1 & Gen 2 & Gen 3 & Gen 4 \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}")


def map_tab(runs, alphas, key):
    head = "Teacher gap & " + " & ".join(f"{round(a * 100)}\\%" for a in alphas) + " \\\\"
    rows = []
    for t in config.TEACHERS + ["base"]:
        if t not in runs:
            continue
        cells = []
        for a in alphas:
            n = config.student_cell(t, a, es=True)["name"]
            m = runs[n]["meta"] if n in runs else None
            v = None if m is None else (m["metrics"]["gap"] if key == "eval" else (m.get("metrics_sampled") or {}).get("gap"))
            cells.append(sg(v, 1))
        rows.append(f"{sg(runs[t]['meta']['metrics']['gap'], 1)}{' (base)' if t == 'base' else ''} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{tabular}{l" + "r" * len(alphas) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")


def rng(lo, hi, d=2):
    return f"{lo:.{d}f}--{hi:.{d}f}"


def mixture_macros(mix, pre, alphas):
    """Weights of audit-style confidence values: list macros and one table block."""
    rows = []
    for lab, L in (("sampled", "S"), ("greedy", "G")):
        vals = [mix[f"{a}:{lab}"]["mean"] for a in alphas if f"{a}:{lab}" in mix]
        put(f"{pre}mix{L}List", ", ".join(f2(v) for v in vals[:-1]) + f" and {f2(vals[-1])}")
        for a in alphas:
            if f"{a}:{lab}" in mix:
                put(f"{pre}mix{L}{WORDS[a]}", f2(mix[f"{a}:{lab}"]["mean"]))
    base = [mix[k]["no_audit_mean"] for k in mix if mix[k]["no_audit_mean"] is not None]
    put(f"{pre}mixBaseMax", f2(max(base)))
    put(f"{pre}mixBaseMaxOne", f2(max(mix[k]["no_audit_max"] for k in mix if mix[k]["no_audit_max"] is not None)))
    put(f"{pre}mixGMaxOne", f2(max(v["max"] for k, v in mix.items() if k.endswith("greedy"))))
    sep = [v["separation"] for v in mix.values()]
    put(f"{pre}mixSepRange", rng(min(sep), max(sep)))
    for a in alphas:
        s_, g_ = mix.get(f"{a}:sampled"), mix.get(f"{a}:greedy")
        if s_ and g_:
            rows.append(f"{round(a * 100)}\\% & {f2(s_['mean'])} ({rng(s_['min'], s_['max'])}) & "
                        f"{f2(g_['mean'])} ({rng(g_['min'], g_['max'])}) & {f2(s_['no_audit_mean'])} / {f2(g_['no_audit_mean'])} \\\\")
    return rows


def record_macros(rm, pre):
    for lab, L in (("greedy", "G"), ("sampled", "S")):
        f = rm.get(f"raw:0.0:{lab}")
        if f:
            put(f"{pre}recGain{L}Zero", f2(f["gain"])); put(f"{pre}recGain{L}ZeroCI", ci(f["gain_ci"]))
        f = rm.get(f"raw7:0.0:{lab}")
        if f:
            put(f"{pre}recSevenGain{L}Zero", f2(f["gain"])); put(f"{pre}recSevenGain{L}ZeroCI", ci(f["gain_ci"]))
        f = rm.get(f"pooled:{lab}")
        if f:
            put(f"{pre}pool{L}Gain", f2(f["gain"])); put(f"{pre}pool{L}Resid", f"{f['resid_sd']:.1f}")
            put(f"{pre}pool{L}Max", f"{f['max_abs_resid']:.1f}"); put(f"{pre}pool{L}N", f["n_cells"])
            put(f"{pre}pool{L}Icept", sg(f["intercept"], 1))
        tr = [v["gain"] for k, v in rm.items() if k.startswith("train:") and k.endswith(lab)]
        if tr:
            put(f"{pre}train{L}Min", f2(min(tr))); put(f"{pre}train{L}Max", f2(max(tr)))


def record_forecast_rows(fr, fam_label):
    rows = []
    for c in fr:
        e, lo_hi = c["error"][-1], c["error_ci"][-1]
        rows.append(f"{fam_label} & {round(c['alpha'] * 100)}\\% & {'new' if c['chain'] == 'fresh' else 'same'} & "
                    f"{'over' if 'p200' in c['start'] else 'under'} & {sg(c['start_record_gap'])} & {sg(c['predicted'][-1])} & "
                    f"{sg(c['observed'][-1])} & {sg(e, 1)} ({sg(lo_hi[0], 1)} to {sg(lo_hi[1], 1)}) \\\\")
    return rows


def record_forecast_macros(fr, pre):
    for c in fr:
        tagn = ("Fresh" if c["chain"] == "fresh" else "") + ("Over" if "p200" in c["start"] else "Under") + ("Zero" if c["alpha"] == 0 else "TwentyFive")
        put(f"{pre}rf{tagn}Err", sg(c["error"][-1], 1))
        put(f"{pre}rf{tagn}Pred", sg(c["predicted"][-1])); put(f"{pre}rf{tagn}Obs", sg(c["observed"][-1]))
    for label in ("same", "fresh"):
        mid = [abs(e) for c in fr if c["alpha"] == 0.25 and c["chain"] == label for e in c["error"]]
        if mid:
            put(f"{pre}rf{'Fresh' if label == 'fresh' else ''}AllMax", f"{max(mid):.1f}")
        early = [abs(e) for c in fr if c["alpha"] == 0.25 and c["chain"] == label for e in c["error"][:2]]
        if early:
            put(f"{pre}rf{'Fresh' if label == 'fresh' else ''}EarlyMax", f"{max(early):.1f}")


def cross_macros(cr, same, pre):
    """pre: student family prefix ('' = Qwen students of Llama teachers, 'llama' = the reverse)."""
    for a, W in ((0.0, "Zero"), (0.75, "SeventyFive")):
        for lab, L in (("greedy", "G"), ("sampled", "S")):
            v = cr.get(f"{a}:{lab}")
            if not v:
                continue
            k = f"{pre}x{L}{W}"
            put(f"{k}Gain", f2(v["gain"])); put(f"{k}GainCI", ci(v["gain_ci"]))
            put(f"{k}Icept", sg(v["intercept"], 1))
            put(f"{k}IceptCI", f"{sg(v['intercept_ci'][0], 1)} to {sg(v['intercept_ci'][1], 1)}")
            put(f"{k}ConfDiff", sg(v["conf_minus_records"], 1))
            put(f"{k}ConfDiffRange", f"{sg(v['conf_minus_records_range'][0], 1)} to {sg(v['conf_minus_records_range'][1], 1)}")
            put(f"{k}AccDiff", f"{abs(v['acc_records_minus_student']):.0f}")
            put(f"{k}GapShift", sg(v["gap_minus_record_gap"], 1))
            put(f"{k}GapRange", f"{sg(v['gap_min'])} to {sg(v['gap_max'])}")
            put(f"{k}AccRange", f"{v['acc_min']:.0f}--{v['acc_max']:.0f}")
            put(f"{k}ConfSlope", f2(v["conf_slope_on_records"]))
            put(f"{k}NQ", v["n_questions"])
            put(f"{k}RecAcc", f"{sum(x['record_acc'] for x in v['rows']) / len(v['rows']):.0f}")
            put(f"{k}N", v["n_teachers"])
            r = {x["teacher"]: x for x in v["rows"]}
            if "teach_p000" in r:
                put(f"{k}CalTeacher", sg(r["teach_p000"]["teacher_gap"])); put(f"{k}CalStudent", sg(r["teach_p000"]["gap"]))
            sf = same.get(f"{a}:{lab}")
            if sf:
                put(f"{pre}same{L}{W}GapRange", f"{sg(sf['gap_min'])} to {sg(sf['gap_max'])}")
                put(f"{pre}same{L}{W}ConfDiff", sg(sf["conf_minus_records"], 1))
                put(f"{pre}same{L}{W}AccRange", f"{sf['acc_min']:.0f}--{sf['acc_max']:.0f}")


def cross_table(cr):
    z, h = cr.get("0.0:greedy"), cr.get("0.75:greedy")
    if not z:
        return None
    hr = {x["teacher"]: x for x in (h["rows"] if h else [])}
    rows = []
    for x in z["rows"]:
        y = hr.get(x["teacher"])
        rows.append(f"{sg(x['teacher_gap'])}{' (base)' if x['teacher'] == 'base' else ''} & {x['record_conf']:.0f} & {x['record_acc']:.0f} & "
                    f"{x['conf']:.0f} & {x['acc']:.0f} & {sg(x['gap'])} & {'--' if not y else sg(y['gap'])} \\\\")
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\nTeacher gap & Records: confidence & Records: accuracy & Student: confidence & "
            "Student: accuracy & Student gap & Student gap, 75\\% audited \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def main():
    S = json.loads((RES / "summary.json").read_text())
    runs = analyze.included(analyze.load())
    e, es = S["extras"], S["es_maps"]
    b = e["base"]
    put("baseConf", f"{b['mean_conf']:.0f}"); put("baseAcc", f"{b['accuracy']:.0f}"); put("baseGap", sg(b["gap"]))
    put("baseAuroc", f2(b["auroc"]))
    tg = [v["gap"] for v in e["teachers"].values()]
    put("teacherGaps", ", ".join(sg(v) for v in tg))
    put("teacherGapMin", sg(min(tg))); put("teacherGapMax", sg(max(tg)))
    put("calTeacherGap", sg(e["teachers"]["teach_p000"]["gap"], 1))
    au = [v["auroc"] for v in e["teachers"].values()]
    put("teacherAurocRange", f"{min(au):.2f}--{max(au):.2f}")
    put("accRange", f"{e['accuracy_range_all_cells'][0]:.0f}--{e['accuracy_range_all_cells'][1]:.0f}")
    put("parsedMin", f"{100 * e['parsed_min']:.1f}")
    put("nRuns", S["n_runs"]); put("nStudentCells", e["n_student_cells"]); put("usdModal", f"{S['usd_total']:.0f}")
    for a, w in WORDS.items():
        for lab, L in (("greedy", "G"), ("sampled", "S")):
            f = es.get(f"{a}:{lab}")
            if f:
                put(f"gain{L}{w}", f2(f["gain"])); put(f"gain{L}{w}CI", ci(f["gain_ci"]))
    put("seedSD", f"{S['noise_floor']['pooled_sd']:.1f}")
    for k, v in e["tag"].items():
        a, tier = k.split(":")
        w = WORDS[float(a)] + tier.capitalize()
        put(f"tagGain{w}", f2(v["gain"])); put(f"tagGain{w}CI", ci(v["gain_ci"]))
        put(f"tagRange{w}", f"{sg(v['y_min'])} to {sg(v['y_max'])}")
    for a, w in WORDS.items():
        for lab, L in (("greedy", "G"), ("sampled", "S")):
            f = es.get(f"{a}:{lab}")
            if f:
                put(f"icept{L}{w}", sg(f["intercept"], 1))
    put("iceptGMin", sg(min(es[f"{a}:greedy"]["intercept"] for a in WORDS), 1))
    put("iceptGMax", sg(max(es[f"{a}:greedy"]["intercept"] for a in WORDS), 1))
    put("iceptSMin", sg(min(es[f"{a}:sampled"]["intercept"] for a in WORDS), 1))
    put("iceptSMax", sg(max(es[f"{a}:sampled"]["intercept"] for a in WORDS), 1))
    for a, v in e["unmarked_seven"].items():
        if v:
            put(f"sevenGain{WORDS[float(a)]}", f2(v["gain"])); put(f"sevenGain{WORDS[float(a)]}CI", ci(v["gain_ci"]))
    for k, v in e["tag_controls"].items():
        lab, key = k.split(":")
        nm = {"probe": "probe", "letter": "letter"}[lab] + {"eval_verified": "A", "eval_unverified": "B", "eval": "Plain"}[key]
        put(f"{nm}Gain", f2(v["gain"])); put(f"{nm}GainCI", ci(v["gain_ci"])); put(f"{nm}N", v["n"])
        put(f"{nm}Range", f"{sg(v['y_min'])} to {sg(v['y_max'])}")
    put("noAuditDrift", f"{e['no_audit_chain_max_drift']:.1f}")
    ns = e["no_audit_seven"]
    put("sevenZeroG", f2(ns["greedy"]["gain"])); put("sevenZeroGCI", ci(ns["greedy"]["gain_ci"]))
    put("sevenZeroS", f2(ns["sampled"]["gain"])); put("sevenZeroSCI", ci(ns["sampled"]["gain_ci"]))
    put("pfiveRight", f2(e["pushback_p5"]["corr_right"])); put("pfiveWrong", f2(e["pushback_p5"]["corr_wrong"]))
    put("parsedMinSampled", f"{100 * e['parsed_min_sampled']:.1f}")
    put("fitNmin", S["es_n_questions"][0]); put("fitNmax", S["es_n_questions"][1])
    rows = []
    for c in S["forecast"]:
        tagn = ("Over" if "p200" in c["start"] else "Under") + ("Zero" if c["alpha"] == 0 else "TwentyFive") + ("G" if c["readout"] == "greedy" else "S")
        put(f"fc{tagn}Err", sg(c["error_gen4"], 1)); put(f"fc{tagn}Pred", sg(c["predicted"][-1])); put(f"fc{tagn}Obs", sg(c["observed"][-1]))
        rows.append(f"{round(c['alpha'] * 100)}\\% & {c['readout']} & {'over' if 'p200' in c['start'] else 'under'} & {sg(c['observed'][0])} & "
                    f"{sg(c['predicted'][-1])} & {sg(c['observed'][-1])} & {sg(c['error_gen4'], 1)} ({sg(c['error_gen4_ci'][0], 1)} to {sg(c['error_gen4_ci'][1], 1)}) \\\\")
    M["tableForecast"] = ("\\begin{tabular}{lllrrrl}\n\\toprule\nAudit share & Readout & Start & Gen 0 & Predicted gen 4 & Observed gen 4 & Error (95\\% interval) \\\\\n\\midrule\n"
                          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    mid = [abs(o - p_) for c in S["forecast"] if c["alpha"] == 0.25 and c["readout"] == "sampled"
           for o, p_ in zip(c["observed"][1:], c["predicted"][1:])]
    put("fcMidMax", f"{max(mid):.1f}")
    fn = [c["n_questions"] for c in S["forecast"]]
    put("fcNmin", min(fn)); put("fcNmax", max(fn))
    ls_ = S["later_steps"]
    put("laterZeroG", sg(ls_["0.0:main:greedy"]["mean_step_toward_zero"], 1))
    put("laterSampG", sg(ls_["0.25:main:greedy"]["mean_step_toward_zero"], 1))
    put("laterSampS", sg(ls_["0.25:main:sampled"]["mean_step_toward_zero"], 1))
    put("laterGreedyG", sg(ls_["0.25:greedy:greedy"]["mean_step_toward_zero"], 1))
    put("laterGreedyS", sg(ls_["0.25:greedy:sampled"]["mean_step_toward_zero"], 1))
    gr = e["greedy_records_alpha0"]
    put("greedyRecGain", f2(gr["gain"])); put("greedyRecGainCI", ci(gr["gain_ci"]))
    pb = e["pushback"]
    put("pushCorrRight", f2(pb["corr_gap_changed_when_right"])); put("pushCorrWrong", f2(pb["corr_gap_changed_when_wrong"]))
    put("pushRightRange", f"{pb['changed_when_right_range'][0]:.0f}--{pb['changed_when_right_range'][1]:.0f}")
    put("baseChangedRight", f"{b['changed_when_right']:.0f}"); put("baseChangedWrong", f"{b['changed_when_wrong']:.0f}")
    ss = e["side_slopes"]
    put("sideUnderZero", f2(ss["0.0:greedy"]["slope_under"])); put("sideOverZero", f2(ss["0.0:greedy"]["slope_over"]))
    put("sideUnderTwentyFive", f2(ss["0.25:greedy"]["slope_under"])); put("sideOverTwentyFive", f2(ss["0.25:greedy"]["slope_over"]))
    # chains
    for r in S["chain_table"]:
        tag = ("Over" if "p200" in r["start"] else "Under") + {(0.0, "main"): "None", (0.25, "main"): "Samp", (0.25, "greedy"): "Greedy"}[(r["alpha"], r["records"])]
        for key, K in (("greedy", "G"), ("sampled", "S")):
            for i, v in enumerate(r[key]):
                put(f"ch{tag}{K}{'ZABCD'[i]}", sg(v))
    M["tableGain"] = gain_table(es, e["tag"], config.ALPHAS_ES)
    M["tableChain"] = chain_tab(S["chain_table"])
    M["tableMapGreedy"] = map_tab(runs, config.ALPHAS_ES, "eval")
    M["tableMapSampled"] = map_tab(runs, config.ALPHAS_ES, "eval_sampled")
    # record-level analyses (existing outputs) and the arms added 2026-11-14 21:00
    mix_rows = mixture_macros(S["mixture"], "", config.ALPHAS_ES[1:])
    record_macros(S["record_maps"], "")
    record_forecast_macros(S["forecast_records"], "")
    fr_rows = record_forecast_rows(S["forecast_records"], "Qwen")
    if S.get("cross"):
        cross_macros(S["cross"], S.get("same_family", {}), "")
        t = cross_table(S["cross"])
        if t:
            M["tableCrossQwen"] = t
    fg = S.get("forge") or {}
    if fg:
        rows = []
        for fs in config.FORGE_SHARES:
            cells = []
            for tier, lab in (("verified", "greedy"), ("verified", "sampled"), ("unverified", "greedy"), ("unverified", "sampled")):
                v = fg.get(f"{fs}:{tier}:{lab}")
                W = {0.0: "Zero", 0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}[fs] + ("V" if tier == "verified" else "U") + ("G" if lab == "greedy" else "S")
                if v:
                    put(f"fg{W}", f2(v["gain"])); put(f"fg{W}CI", ci(v["gain_ci"]))
                    put(f"fg{W}Range", f"{sg(v['y_min'])} to {sg(v['y_max'])}")
                    if v["predicted_gain"] is not None:
                        put(f"fg{W}Pred", f2(v["predicted_gain"]))
                    pr = "" if (v["predicted_gain"] is None or tier == "unverified") else f" [{f2(v['predicted_gain'])}]"
                    cells.append(f"{f2(v['gain'])} ({ci(v['gain_ci'])}){pr}")
                else:
                    cells.append("--")
            if any(c != "--" for c in cells):
                share = config.FORGE_ALPHA / (config.FORGE_ALPHA + fs)
                rows.append(f"{round(fs * 100)}\\% & {round(100 * share)}\\% & " + " & ".join(cells) + " \\\\")
        M["tableForge"] = ("\\begin{tabular}{rrcccc}\n\\toprule\nForged & Audited among tagged & `verified', greedy & `verified', sampled & "
                           "`unverified', greedy & `unverified', sampled \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    fq = S.get("fq_table") or []
    if fq and all(v is not None for r in fq for v in r["greedy"]):
        rows = []
        for r in fq:
            start = "over" if "p200" in r["start"] else "under"
            arm = "No audit" if r["alpha"] == 0 else "25\\%, sampled records"
            cells = " & ".join(f"{sg(g)} / {'--' if s_ is None else sg(s_)}" for g, s_ in zip(r["greedy"], r["sampled"]))
            same = " & ".join(f"{sg(g)} / {'--' if s_ is None else sg(s_)}" for g, s_ in zip(r["same_greedy"], r["same_sampled"]))
            rows.append(f"{arm} & {start} & new & {cells} \\\\")
            rows.append(f" &  & same & {same} \\\\")
            tagn = ("Over" if start == "over" else "Under") + ("None" if r["alpha"] == 0 else "Samp")
            for key, K in (("greedy", "G"), ("sampled", "S")):
                for i, v in enumerate(r[key]):
                    put(f"fq{tagn}{K}{'ZABCD'[i]}", sg(v))
            d = [abs(v) for v in r["diff_greedy"][2:] + r["diff_sampled"][2:] if v is not None]
            put(f"fq{tagn}MaxDiff", f"{max(d):.1f}")
            put(f"fq{tagn}GenOneDiff", f"{max(abs(r['diff_greedy'][1]), abs(r['diff_sampled'][1])):.1f}")
        alld = [abs(v) for r in fq for v in r["diff_greedy"][2:] + r["diff_sampled"][2:] if v is not None]
        put("fqMaxDiff", f"{max(alld):.1f}"); put("fqMeanDiff", f"{sum(alld) / len(alld):.1f}")
        put("fqGenOneDiff", f"{max(max(abs(r['diff_greedy'][1]), abs(r['diff_sampled'][1])) for r in fq):.1f}")
        zero = [abs(v - r["greedy"][0]) for r in fq if r["alpha"] == 0 for v in r["greedy"]]
        put("fqNoAuditDrift", f"{max(zero):.1f}")
        put("chainDriftAll", f"{max(max(zero), e['no_audit_chain_max_drift']):.1f}")
        M["tableFq"] = ("\\begin{tabular}{lllccccc}\n\\toprule\nArm & Start & Questions & Gen 0 & Gen 1 & Gen 2 & Gen 3 & Gen 4 \\\\\n\\midrule\n"
                        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # scorer audit
    sa = RES / "scoring_audit.json"
    if sa.exists():
        A = json.loads(sa.read_text())
        put("scoreAgree", f"{100 * min(v['agreement'] for v in A.values()):.0f}")
        put("scoreOffset", f"{sum(v['gap_rule'] - v['gap_llm'] for v in A.values()) / len(A):.1f}")
        put("scoreRuns", len(A))
        offs = [v["gap_rule"] - v["gap_llm"] for v in A.values()]
        put("scoreOffsetRange", f"{min(offs):.1f}--{max(offs):.1f}")
        ag = [100 * v["agreement"] for v in A.values()]
        put("scoreAgreeRange", f"{min(ag):.1f}--{max(ag):.1f}")
    # hosted panel
    pp = RES / "panel.json"
    if pp.exists():
        P = json.loads(pp.read_text())
        inc = {k: v for k, v in P.items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
        gaps = sorted(v["gap"] for v in inc.values())
        put("panelTested", len(P)); put("panelIncluded", len(inc))
        put("panelOver", sum(g > 0 for g in gaps)); put("panelGapMin", sg(gaps[0])); put("panelGapMax", sg(gaps[-1]))
        put("panelGapMedian", sg(gaps[len(gaps) // 2] if len(gaps) % 2 else (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2))
        put("panelConfMin", f"{min(v['mean_conf'] for v in inc.values()):.0f}")
        rows = [f"{k.split('/')[-1].replace('_', chr(92) + '_')} & {v['accuracy']:.0f} & {v['mean_conf']:.0f} & {sg(v['gap'])} & {100 * v['parsed']:.0f} \\\\"
                for k, v in sorted(inc.items(), key=lambda kv: -kv[1]["gap"])]
        M["tablePanel"] = ("\\begin{tabular}{lrrrr}\n\\toprule\nModel & Accuracy & Stated confidence & Gap & In format (\\%) \\\\\n\\midrule\n"
                           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
        put("panelExcluded", ", ".join(f"{k.split('/')[-1].replace('_', chr(92) + '_')} ({100 * v.get('parsed', 0):.0f}\\%)" for k, v in sorted(P.items()) if k not in inc) or "none")
    # second family
    lp, ls = RES / "runs_llama31.json", RES / "summary_llama31.json"
    if lp.exists() and ls.exists():
        L = json.loads(ls.read_text())
        lr = analyze.included(json.loads(lp.read_text())["runs"])
        lb = lr["base"]["meta"]["metrics"]
        put("llamaBaseConf", f"{lb['mean_conf']:.0f}"); put("llamaBaseAcc", f"{lb['accuracy']:.0f}")
        put("llamaBaseGap", sg(lb["gap"])); put("llamaBaseParsed", f"{100 * lb['parsed']:.1f}")
        lt = [lr[t]["meta"]["metrics"]["gap"] for t in config.TEACHERS if t in lr]
        if lt:
            put("llamaTeacherGapMin", sg(min(lt))); put("llamaTeacherGapMax", sg(max(lt)))
        for a in config.ADDON_ALPHAS:
            for lab, K in (("greedy", "G"), ("sampled", "S")):
                f = L["es_maps"].get(f"{a}:{lab}")
                if f:
                    put(f"llamaGain{K}{WORDS[a]}", f2(f["gain"])); put(f"llamaGain{K}{WORDS[a]}CI", ci(f["gain_ci"]))
        M["tableLlama"] = gain_table(L["es_maps"], None, config.ADDON_ALPHAS)
        M["tableLlamaChain"] = chain_tab(L["chain_table"])
        put("llamaChainGens", min(sum(v is not None for v in r["greedy"]) for r in L["chain_table"]) - 1)
        for r in L["chain_table"]:
            tg = ("Over" if "p200" in r["start"] else "Under") + {(0.0, "main"): "None", (0.25, "main"): "Samp", (0.25, "greedy"): "Greedy"}[(r["alpha"], r["records"])]
            last = [v for v in r["greedy"] if v is not None]
            put(f"lch{tg}Start", sg(last[0])); put(f"lch{tg}End", sg(last[-1]))
        for c in L["forecast"]:
            tagn = ("Over" if "p200" in c["start"] else "Under") + ("Zero" if c["alpha"] == 0 else "TwentyFive") + ("G" if c["readout"] == "greedy" else "S")
            put(f"lfc{tagn}Err", sg(c["error_gen4"], 1))
        lfn = [c["n_questions"] for c in L["forecast"]]
        put("llamaFcNmin", min(lfn)); put("llamaFcNmax", max(lfn))
        put("llamaStudents", sum(k.startswith("stu__") for k in lr)); 
        put("llamaLaterZeroG", sg(L["later_steps"]["0.0:main:greedy"]["mean_step_toward_zero"], 1))
        put("llamaFitNmin", L["es_n_questions"][0]); put("llamaFitNmax", L["es_n_questions"][1])
        put("llamaRuns", L["n_runs"]); put("llamaUsd", f"{L['usd_total']:.0f}")
        # every Llama run recorded "L40S" and the L40S rate (GPU env var unset in the containers);
        # cells after 16:52 CET ran on A100-80GB, so the true cost lies between the two rates.
        lsec = sum(r["meta"].get("seconds", 0) for r in json.loads(lp.read_text())["runs"].values())
        put("llamaUsdHigh", f"{lsec * 2.50 / 3600:.0f}")
        put("nStudentsAll", int(M["nStudentCells"]) + int(M["llamaStudents"]))
        if L.get("mixture"):
            mix_rows_l = mixture_macros(L["mixture"], "llama", config.ADDON_ALPHAS[1:])
            M["tableMixture"] = ("\\begin{tabular}{llccc}\n\\toprule\nModel & Audit share & Sampled answer & Greedy answer & No-audit students (sampled / greedy) \\\\\n\\midrule\n"
                                 + "\n".join("Qwen2.5-7B & " + r for r in mix_rows) + "\n\\midrule\n"
                                 + "\n".join("Llama-3.1-8B & " + r for r in mix_rows_l) + "\n\\bottomrule\n\\end{tabular}")
            record_macros(L["record_maps"], "llama")
            record_forecast_macros(L["forecast_records"], "llama")
            fr_rows += record_forecast_rows(L["forecast_records"], "Llama")
        if L.get("cross"):
            cross_macros(L["cross"], L.get("same_family", {}), "llama")
            t = cross_table(L["cross"])
            if t:
                M["tableCrossLlama"] = t
    # third family (Qwen3-8B, thinking off): reduced grid, no cross cells, no chains
    qp, qs = RES / "runs_qwen3.json", RES / "summary_qwen3.json"
    if qp.exists() and qs.exists():
        Q = json.loads(qs.read_text())
        qr = analyze.included(json.loads(qp.read_text())["runs"])
        qb = qr["base"]["meta"]["metrics"]
        put("qthreeBaseConf", f"{qb['mean_conf']:.0f}"); put("qthreeBaseAcc", f"{qb['accuracy']:.0f}")
        put("qthreeBaseGap", sg(qb["gap"]))
        qt = [qr[t]["meta"]["metrics"]["gap"] for t in config.TEACHERS if t in qr]
        put("qthreeTeacherGapMin", sg(min(qt))); put("qthreeTeacherGapMax", sg(max(qt)))
        for a in config.ADDON_ALPHAS:
            for lab, K in (("greedy", "G"), ("sampled", "S")):
                f = Q["es_maps"].get(f"{a}:{lab}")
                if f:
                    put(f"qthreeGain{K}{WORDS[a]}", f2(f["gain"])); put(f"qthreeGain{K}{WORDS[a]}CI", ci(f["gain_ci"]))
        M["tableQwenThree"] = gain_table(Q["es_maps"], None, config.ADDON_ALPHAS)
        put("qthreeStudents", sum(k.startswith("stu__") for k in qr))
        put("qthreeRuns", Q["n_runs"]); put("qthreeUsd", f"{Q['usd_total']:.0f}")
        put("nStudentsAll", int(M["nStudentsAll"]) + int(M["qthreeStudents"]))
        record_macros(Q["record_maps"], "qthree")
        if Q.get("mixture"):
            mix_rows_q = mixture_macros(Q["mixture"], "qthree", config.ADDON_ALPHAS[1:])
            M["tableMixture"] = M["tableMixture"].replace(
                "\n\\bottomrule", "\n\\midrule\n" + "\n".join("Qwen3-8B & " + r for r in mix_rows_q) + "\n\\bottomrule")
    M["tableRecordForecast"] = ("\\begin{tabular}{lllllrrl}\n\\toprule\nModel & Audit share & Questions & Start & Start (records) & Predicted gen 4 & Observed gen 4 & Error (95\\% interval) \\\\\n\\midrule\n"
                                + "\n".join(fr_rows) + "\n\\bottomrule\n\\end{tabular}")
    (Path(__file__).parent / "numbers.tex").write_text(
        "% generated by paper/make_numbers.py from results/*.json -- do not edit\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    print(f"{len(M)} macros")
    for k, v in M.items():
        if not k.startswith("table"):
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
