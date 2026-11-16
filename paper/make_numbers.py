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
    if v is None:
        return "n/a"
    v = 0.0 if round(v, d) == 0 else v  # no "-0.0"
    return f"{v:+.{d}f}".replace("-", "$-$")


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


def distill_macros():
    """Distillation arm (results/dist_cells.json from loopgain/distill.py report). Macro names:
    d + student (Q = Qwen3.5-9B, G = Granite-4.1-8B) + cell + quantity; sampled answer unless Greedy."""
    f = RES / "dist_cells.json"
    if not f.exists():
        return
    cells = json.loads(f.read_text())
    labels = json.loads((RES / "dist" / "labels.json").read_text())
    FAM = {"qwen35": ("Q", "Qwen"), "granite41": ("G", "Gran.")}
    TEACH = {"ds41f": ("", "DS"), "q38_27b": ("M", "Q3.8")}
    SETS = (("raw", "Raw", "own"), ("tcal", "Tcal", "t-cal."),
            ("scal", "Scal", "s-fit"), ("shuf", "Shuf", "shuf."))

    def get(fam, tname, readout="sampled", seed=1):
        c = [r for r in cells if r["family"] == fam and r["teacher"] == tname and r["readout"] == readout
             and r["seed"] == seed]
        return c[0] if c else None

    for t, (tp, _) in TEACH.items():
        li = labels[t]
        raw = li["sets"][f"{t}__raw"]
        put(f"d{tp}TeacherAcc", f"{raw['teacher_accuracy']:.0f}"); put(f"d{tp}TeacherConf", f"{raw['label_mean']:.0f}")
        put(f"d{tp}TeacherGap", sg(raw["label_mean"] - raw["teacher_accuracy"]))
        put(f"d{tp}TeacherParsed", f"{100 * li['parsed']:.1f}"); put(f"d{tp}TeacherKept", f"{raw['n']:,}".replace(",", "{,}"))
        put(f"d{tp}TcalLabel", f"{li['sets'][f'{t}__tcal']['label_mean']:.0f}")
    rows, allc, parsed = [], [], []
    for fam, (fp, fname) in FAM.items():
        for t, (tp, tname) in TEACH.items():
            for key, K, desc in SETS:
                name = f"{t}__{key}_{fam}" if key == "scal" else f"{t}__{key}"
                c, g = get(fam, name), get(fam, name, "greedy")
                if not c:
                    continue
                pre = f"d{fp}{tp}{K}"
                put(pre + "Gap", sg(c["gap"])); put(pre + "GapCI", f"{sg(c['gap_lo'])} to {sg(c['gap_hi'])}")
                put(pre + "Acc", f"{c['accuracy']:.0f}"); put(pre + "Conf", f"{c['conf']:.0f}")
                put(pre + "Label", f"{c['label_mean']:.0f}"); put(pre + "Auroc", f2(c["auroc"]))
                put(pre + "Ece", f"{c['ece']:.1f}")
                put(pre + "GreedyGap", sg(g["gap"]))
                if c.get("recal"):
                    put(pre + "Recal", sg(c["recal"]["gap"], 1)); put(pre + "GreedyRecal", sg(g["recal"]["gap"], 1))
                seeds = [x["gap"] for s_ in (1, 2, 3) if (x := get(fam, name, seed=s_))]
                if len(seeds) == 3:
                    put(pre + "Seeds", f"{sg(min(seeds))} to {sg(max(seeds))}")
                allc.append(abs(c["conf_minus_label"])); parsed += [c["parsed"], g["parsed"]]
                rows.append(f"{fname} & {tname} & {desc} & {c['label_mean']:.1f} & {c['conf']:.1f} & {c['accuracy']:.1f} & "
                            f"{sg(c['gap'], 1)} & {c['gap_lo']:.0f}..{c['gap_hi']:.0f} & {sg(g['gap'], 1)} & "
                            f"{f2(c['auroc'])} & {c['ece']:.1f} & {sg(c['recal']['gap'], 1) if c.get('recal') else '--'} \\\\")
        c, g = get(fam, "base"), get(fam, "base", "greedy")
        if c:
            pre = f"d{fp}Own"
            put(pre + "Gap", sg(c["gap"], 1)); put(pre + "Acc", f"{c['accuracy']:.0f}"); put(pre + "Auroc", f2(c["auroc"]))
            put(pre + "GreedyGap", sg(g["gap"])); put(pre + "Recal", sg(c["recal"]["gap"], 1))
            parsed += [c["parsed"], g["parsed"]]
            rows.append(f"{fname} & itself & audit & {c['label_mean']:.1f} & {c['conf']:.1f} & "
                        f"{c['accuracy']:.1f} & {sg(c['gap'], 1)} & {c['gap_lo']:.0f}..{c['gap_hi']:.0f} & "
                        f"{sg(g['gap'], 1)} & {f2(c['auroc'])} & {c['ece']:.1f} & {sg(c['recal']['gap'], 1)} \\\\")
    own = {fam: get(fam, "base")["accuracy"] for fam in FAM if get(fam, "base")}
    shift = [r["accuracy"] - own[r["family"]] for r in cells if r["readout"] == "sampled" and r["seed"] == 1
             and r["teacher"].startswith("ds41f") and r["family"] in own]
    put("dAccShiftMax", f"{max(shift):.1f}"); put("dAccShiftMin", sg(min(shift), 1))
    runs_usd = 0.0
    for fam in FAM:
        rf = RES / f"runs_dist_{fam}.json"
        if rf.exists():
            runs_usd += sum(v["meta"].get("usd", 0) for v in json.loads(rf.read_text())["runs"].values())
    put("dUsd", f"{runs_usd:.0f}")
    put("dCopyMax", f"{max(allc):.1f}"); put("dParsedMin", f"{100 * min(parsed):.1f}")
    put("dCells", len({(r['family'], r['cell']) for r in cells}))
    recal = [abs(r["recal"]["gap"]) for r in cells if r.get("recal") and r["readout"] == "sampled"]
    put("dRecalMax", f"{max(recal):.1f}")
    accs = [r["accuracy"] for r in cells if r["readout"] == "sampled" and r["teacher"].startswith("ds41f")]
    put("dStudentAccRange", f"{min(accs):.0f}--{max(accs):.0f}")
    # greedy accuracy and stated confidence across the four DeepSeek label sets (answers identical)
    for fam, (fp, _) in FAM.items():
        g = [get(fam, f"ds41f__{s}" + (f"_{fam}" if s == "scal" else ""), "greedy") for s in ("raw", "tcal", "scal", "shuf")]
        put(f"d{fp}GreedyAccRange", f"{min(r['accuracy'] for r in g):.1f}--{max(r['accuracy'] for r in g):.1f}")
        put(f"d{fp}GreedyConfRange", f"{min(r['conf'] for r in g):.0f} to {max(r['conf'] for r in g):.0f}")
    M["tableDistill"] = ("\\begin{tabular}{lllrrrrlrrrr}\n\\toprule\nStud. & From & Labels & Label & Stated & "
                         "Acc. & Gap & 95\\% & Gr. & AUC & ECE & Fixed \\\\\n\\midrule\n"
                         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # Decomposition of the student's gap: copying error + label gap + accuracy difference.
    for fam, (fp, _) in FAM.items():
        for key, K in (("raw", "Raw"), ("tcal", "Tcal")):
            c = get(fam, f"ds41f__{key}")
            if c:
                AT = c["record_accuracy"]
                put(f"d{fp}{K}Copy", sg(c["conf"] - c["label_mean"], 1)); put(f"d{fp}{K}LabelGap", sg(c["label_mean"] - AT, 1))
                put(f"d{fp}{K}AccDiff", sg(AT - c["accuracy"], 1)); put(f"d{fp}{K}GapOne", sg(c["gap"], 1))
    # Concrete readings from the greedy answers (what a user sees at temperature zero).
    EXQ = "Who wrote the opera A Masked Ball?"
    for fam, (fp, _) in FAM.items():
        rf = RES / f"runs_dist_{fam}.json"
        if not rf.exists():
            continue
        d = json.loads(rf.read_text())
        def ev(tname):
            for r in d["runs"].values():
                if r["meta"]["teacher"] == tname and r["meta"]["seed"] == 1:
                    return {x["id"]: x for x in r["eval"]["eval"] if x["answer"] is not None and x["conf"] is not None}
        for K, tname in (("Raw", "ds41f__raw"), ("Tcal", "ds41f__tcal"), ("Scal", f"ds41f__scal_{fam}"), ("Own", "base")):
            w = [x for x in ev(tname).values() if not x["correct"]]
            put(f"d{fp}{K}Wrong", len(w))
            put(f"d{fp}{K}WrongHigh", f"{100 * sum(x['conf'] >= 85 for x in w) / len(w):.0f}")
        if fam == "qwen35":
            qid = [q["id"] for q in d["data"]["E"] if q["q"] == EXQ][0]
            ans = {ev(t)[qid]["answer"] for t in ("ds41f__raw", "ds41f__tcal", "ds41f__scal_qwen35")}
            assert ans == {"Giacomo Puccini"} and not ev("ds41f__raw")[qid]["correct"], ans
            put("dExRaw", f"{ev('ds41f__raw')[qid]['conf']:.0f}"); put("dExTcal", f"{ev('ds41f__tcal')[qid]['conf']:.0f}")
            put("dExScal", f"{ev('ds41f__scal_qwen35')[qid]['conf']:.0f}")
    st = RES / "dist_stats.json"
    if st.exists():
        S = json.loads(st.read_text())
        def d3(v):
            return f"{sg(v['diff'], 2)} ({sg(v['lo'], 2)} to {sg(v['hi'], 2)})"
        gr = []
        for fam, (fp, _) in FAM.items():
            o = S[fam]
            put(f"d{fp}AurocVsOwn", d3(o["auroc:ds41f__tcal-own"])); put(f"d{fp}AurocVsShuf", d3(o["auroc:tcal-shuf"]))
            put(f"d{fp}MAurocVsOwn", d3(o["auroc:q38_27b__tcal-own"]))
            v = o["acc_greedy:ds41f__tcal-own"]
            put(f"d{fp}AccUp", f"{sg(v['diff'], 1)} ({sg(v['lo'], 1)} to {sg(v['hi'], 1)})")
            gr += [o[f"acc_greedy:ds41f__{lab}-own"]["diff"] for lab in ("raw", "tcal", f"scal_{fam}")]
            for key, nm in (("acc_greedy:raw-scal", "AccSliderGreedy"), ("acc:raw-scal", "AccSliderSampled")):
                if key in o:
                    w = o[key]
                    put(f"d{fp}{nm}", f"{sg(w['diff'], 1)} ({sg(w['lo'], 1)} to {sg(w['hi'], 1)})")
            for key, nm in (("sel50:tcal-shuf", "Shuf"), ("sel50:tcal-own", "Own")):
                w = o[key]
                put(f"d{fp}SelVs{nm}", f"{sg(w['diff'], 1)} ({sg(w['lo'], 1)} to {sg(w['hi'], 1)})")
                put(f"d{fp}Sel{nm}", f"{w['b']:.0f}")
            put(f"d{fp}SelTcal", f"{o['sel50:tcal-shuf']['a']:.0f}")
            v = o["acc_greedy:q38_27b__tcal-own"]
            put(f"d{fp}MAccUp", sg(v["diff"], 1))
        put("dAccUpRange", f"{min(gr):.1f} to {max(gr):.1f}")
        put("dAccDiff", f"{float(M['dTeacherAcc']) - (float(M['dQOwnAcc']) + float(M['dGOwnAcc'])) / 2:.0f}")
    rg = RES / "dist_regrade.json"
    if rg.exists():
        r = json.loads(rg.read_text())
        ag = [100 * v["agreement"] for v in r.values()]
        put("dHaikuAgree", f"{min(ag):.1f}--{max(ag):.1f}")
        for fam, (fp, _) in FAM.items():
            for key, K in ((f"student:{fam}:ds41f__tcal", "Tcal"), (f"student:{fam}:ds41f__scal_{fam}", "Scal")):
                if key in r:
                    put(f"d{fp}{K}HaikuGap", sg(r[key]["gap_llm"]))
                    put(f"d{fp}{K}HaikuGapOne", sg(r[key]["gap_llm"], 1))
        for t, (tp, _) in TEACH.items():
            if f"teacher:{t}" in r:
                put(f"d{tp}TeacherHaikuAcc", f"{r[f'teacher:{t}']['acc_llm']:.0f}")


def card_macros():
    """Confidence card (main text): hosted models, the distilled students and a human reference, with a class
    from fixed bands on the gap. Hosted rows: greedy answer; student rows: sampled answer, seed 1."""
    pf, cf = RES / "panel.json", RES / "dist_cells.json"
    if not (pf.exists() and cf.exists()):
        return
    P = {k: v for k, v in json.loads(pf.read_text()).items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
    cells = json.loads(cf.read_text())

    def band(g):
        a = abs(g)
        if a < 5:
            return "calibrated"
        return ("strongly " if a >= 15 else "") + ("over" if g > 0 else "under")

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2

    def z(g):
        return "0" if abs(g) < 0.5 else sg(g)

    rows = []
    teacher = [v for k, v in P.items() if "DeepSeek-V4.1-Flash" in k][0]
    low = min(P.values(), key=lambda v: v["accuracy"])
    mg = med([v["gap"] for v in P.values()])
    for name, a, c, g, note in (
            ("Teacher (DeepSeek)", teacher["accuracy"], teacher["mean_conf"], teacher["gap"], ""),
            (f"Median of {len(P)} hosted", med([v["accuracy"] for v in P.values()]),
             med([v["mean_conf"] for v in P.values()]), mg, ""),
            ("Least accurate hosted", low["accuracy"], low["mean_conf"], low["gap"], "")):
        rows.append(f"{name} & {a:.0f} & {c:.0f} & {sg(g)} & {band(g)}{note} \\\\")
    rows.append("\\midrule")

    def get(fam, tname):
        return [r for r in cells if r["family"] == fam and r["teacher"] == tname and r["readout"] == "sampled"
                and r["seed"] == 1][0]
    for name, tn, note in (("own records", "base", ""),
                           ("teacher's records", "ds41f__raw", ""),
                           ("teacher calibrated", "ds41f__tcal", ""),
                           ("fitted to student", "ds41f__scal_{fam}", "")):
        q, g = get("qwen35", tn.format(fam="qwen35")), get("granite41", tn.format(fam="granite41"))
        assert band(q["gap"]) == band(g["gap"]), (name, q["gap"], g["gap"])
        rows.append(f"Students: {name} & {q['accuracy']:.0f} / {g['accuracy']:.0f} & {q['conf']:.0f} / {g['conf']:.0f} & "
                    f"{z(q['gap'])} / {z(g['gap'])} & {band(q['gap'])}{note} \\\\")
    M["tableCard"] = ("\\begin{tabular}{p{0.34\\linewidth}p{0.13\\linewidth}p{0.13\\linewidth}p{0.14\\linewidth}p{0.18\\linewidth}}\n\\toprule\nWho & Is right (\\%) & Says (\\%) & Gap & Class \\\\\n\\midrule\n"
                      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def late_macros():
    """Arms added after the first submission-ready version (PROTOCOL.md, 15 Nov 23:51 onwards):
    k = ask arm (results/ask_stats.json), m = multiple-choice arm (results/mc_stats.json),
    u = audit-size arm (results/aud_labels.json, results/aud_stats.json). Sampled answer."""
    fams = (("qwen35", ("Q", "Qwen")), ("granite41", ("G", "Gran.")))
    held = n = 0
    f = RES / "ask_stats.json"
    if f.exists():
        S, rows = json.loads(f.read_text()), []
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            c = {k.split(":")[0]: v for k, v in S[fam]["cells"].items() if k.endswith(":sampled")}
            cr = S[fam]["carry"]["seed1:sampled"]
            for w, W in (("F1", "Fone"), ("F2", "Ftwo")):
                put(f"k{fp}Carry{W}", f2(cr[w]["c"])); put(f"k{fp}Diff{W}", sg(cr[w]["d"]))
                for k, K in (("tcal", "Tcal"), ("scal", "Scal"), ("shuf", "Shuf"), ("untrained", "Base")):
                    put(f"k{fp}{W}{K}Gap", sg(c[k][w]["gap"])); put(f"k{fp}{W}{K}Conf", f"{c[k][w]['conf']:.0f}")
                    put(f"k{fp}{W}{K}Auroc", f2(c[k][w]["auroc"]))
            put(f"k{fp}DiffT", sg(cr["T"]["d"]))
            for k, lab in (("tcal", "t-cal."), ("scal", "s-fit"), ("shuf", "shuf."), ("own", "own records"),
                           ("tcal_s2", "t-cal., seed 2"), ("scal_s2", "s-fit, seed 2"), ("untrained", "untrained")):
                e = c[k]
                rows.append(f"{fname} & {lab} & " + " & ".join(
                    f"{e[w]['conf']:.1f} & {e[w]['accuracy']:.1f} & {sg(e[w]['gap'], 1)} & {f2(e[w]['auroc'])}" for w in ("T", "F1", "F2"))
                    + f" & {100 * min(e['F1']['in_format'], e['F2']['in_format']):.0f} \\\\")
            held += sum(S[fam]["predictions"].values()); n += len(S[fam]["predictions"])
            put(f"k{fp}Usd", f"{S[fam]['usd']:.2f}")
        put("kPredHeld", held); put("kPredAll", n)
        M["tableAsk"] = ("\\begin{tabular}{llrrrrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{Trained format} & \\multicolumn{4}{c}{Reworded} & "
                         "\\multicolumn{4}{c}{Second turn} & \\\\\nStud. & Labels & Says & Acc. & Gap & AUC & Says & Acc. & Gap & AUC & Says & Acc. & Gap & AUC & Fmt \\\\\n\\midrule\n"
                         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f = RES / "mc_stats.json"
    if f.exists():
        S, rows, mh, mn = json.loads(f.read_text()), [], 0, 0
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            c = S[fam]["cells"]
            put("mTeacherAcc", f"{S[fam]['teacher_record_accuracy']:.1f}")
            for k, K in (("raw", "Raw"), ("tcal", "Tcal"), ("scal", "Scal"), ("shuf", "Shuf"), ("untrained", "Base")):
                if k not in c:
                    continue
                e = c[k]["sampled"]
                put(f"m{fp}{K}Gap", sg(e["gap"])); put(f"m{fp}{K}Conf", f"{e['conf']:.0f}"); put(f"m{fp}{K}Acc", f"{e['accuracy']:.0f}")
                put(f"m{fp}{K}Auroc", f2(e["auroc"])); put(f"m{fp}{K}Sel", f"{e['sel50']:.0f}")
                put(f"m{fp}{K}GapCI", f"{sg(e['gap_lo'])} to {sg(e['gap_hi'])}")
            if "scal" in c:
                put(f"m{fp}ScalLabel", f"{c['scal']['label']:.0f}")
            for k, lab in (("raw", "own"), ("tcal", "t-cal."), ("scal", "s-fit"), ("shuf", "shuf."),
                           ("tcal_s2", "t-cal., seed 2"), ("scal_s2", "s-fit, seed 2"), ("untrained", "untrained")):
                if k not in c:
                    continue
                e, g = c[k]["sampled"], c[k]["greedy"]
                lab_mean = "--" if c[k]["label"] is None else f"{c[k]['label']:.1f}"
                rows.append(f"{fname} & {lab} & {lab_mean} & {e['conf']:.1f} & {e['accuracy']:.1f} & {sg(e['gap'], 1)} "
                            f"({sg(e['gap_lo'], 0)} to {sg(e['gap_hi'], 0)}) & {sg(g['gap'], 1)} & {e['ece']:.1f} & {f2(e['auroc'])} & "
                            f"{e['sel50']:.0f} & {100 * e['parsed']:.1f} & {e['accuracy_all']:.1f} \\\\")
            if "auroc:tcal-shuf" in S[fam]:
                a, h = S[fam]["auroc:tcal-shuf"], S[fam]["sel50:tcal-shuf"]
                put(f"m{fp}AurocVsShuf", f"{sg(a['d'], 2)} ({sg(a['lo'], 2)} to {sg(a['hi'], 2)})")
                put(f"m{fp}SelVsShuf", f"{sg(h['d'])} ({sg(h['lo'])} to {sg(h['hi'])})")
            mh += sum(S[fam]["predictions"].values()); mn += len(S[fam]["predictions"])
            put(f"m{fp}Usd", f"{sum(v.get('usd') or 0 for v in c.values()):.2f}")
        g = RES / "mc_recal.json"  # held-out correction of the trained students, each half corrected by the other (post hoc)
        if g.exists():
            R, worst = json.loads(g.read_text()), 0.0
            for fam, (fp, fname) in fams:
                if fam not in R:
                    continue
                for k, K in (("scal", "Scal"), ("tcal", "Tcal")):
                    e = R[fam]["cells"][k]
                    put(f"m{fp}{K}RecalHalves", f"{sg(e['gap_half_a'], 1)} and {sg(e['gap_half_b'], 1)}")
                    put(f"m{fp}{K}RecalAuroc", f2(e["auroc_after"]))
                worst = max([worst] + [abs(e[h]) for e in R[fam]["cells"].values() for h in ("gap_half_a", "gap_half_b")])
                b = json.loads((RES / "mc" / f"basemc_{fam}.json").read_text())
                put(f"m{fp}BaseEvalAcc", f"{b['metrics_sampled']['accuracy']:.0f}"); put(f"m{fp}BaseRecAcc", f"{b['d_main']['accuracy']:.0f}")
            put("mRecalMaxHalf", f"{worst:.1f}")
        put("mPredHeld", mh); put("mPredAll", mn)
        held += mh; n += mn
        M["tableMc"] = ("\\begin{tabular}{llrrrlrrrrrr}\n\\toprule\nStud. & Labels & Label & Says & Acc. & Gap (95\\%) & Gr. & ECE & AUC & Half & Fmt & Acc.* \\\\\n\\midrule\n"
                        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f, g = RES / "aud_labels.json", RES / "aud_stats.json"
    if f.exists():
        L, rows, uh, un = json.loads(f.read_text()), [], 0, 0
        A = json.loads(g.read_text()) if g.exists() else {}
        for fam, (fp, fname) in fams:
            for ns, d in L[fam]["sizes"].items():
                W = {"50": "Fifty", "100": "Hundred", "250": "TwoFifty", "500": "FiveHundred"}[ns]
                put(f"u{fp}Within{W}", f"{100 * d['draws_within5']:.0f}")
                cell = (A.get(fam, {}).get("cells") or {}).get(ns)
                tr = "-- & -- & -- & --"
                if cell:
                    e = cell["sampled"]
                    tr = f"{e['conf']:.1f} & {e['accuracy']:.1f} & {sg(e['gap'], 1)} ({sg(e['gap_lo'], 0)} to {sg(e['gap_hi'], 0)}) & {f2(e['auroc'])}"
                    put(f"u{fp}Gap{W}", sg(e["gap"])); put(f"u{fp}Auroc{W}", f2(e["auroc"]))
                rows.append(f"{fname} & {ns} & {d['draws_med']:.1f} ({d['draws_lo']:.0f} to {d['draws_hi']:.0f}) & {100 * d['draws_within5']:.0f} & "
                            f"{d['label_mean']:.1f} & {tr} \\\\")
            if fam in A:
                a = A[fam]["all_records"]
                rows.append(f"{fname} & 1{{,}}978 & -- & -- & {L[fam]['full_label_mean']:.1f} & {a['conf']:.1f} & {a['accuracy']:.1f} & {sg(a['gap'], 1)} & {f2(a['auroc'])} \\\\")
                uh += sum(A[fam]["predictions"].values()); un += len(A[fam]["predictions"])
                put(f"u{fp}Usd", f"{sum(v.get('usd') or 0 for v in A[fam]['cells'].values()):.2f}")
        if un:
            put("uPredHeld", uh); put("uPredAll", un)
        held += uh; n += un
        M["tableAud"] = ("\\begin{tabular}{lrlrrrrlr}\n\\toprule\n & & \\multicolumn{2}{c}{500 random draws} & \\multicolumn{5}{c}{One draw, trained student} \\\\\n"
                         "Stud. & Checked & Label (95\\%) & Within 5 & Label & Says & Acc. & Gap (95\\%) & AUC \\\\\n\\midrule\n"
                         + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f = RES / "mc" / "ranking_diagnosis.json"
    if f.exists():
        D, rows = json.loads(f.read_text()), []
        for task, d in D.items():
            for k, lab in (("raw", "raw"), ("tcal", "calibrated")):
                rows.append(f"{task} & {lab} & {d['n']:,} & {f2(d['teacher_self'][k])} & {f2(d['qwen35'][k])} & {f2(d['granite41'][k])} \\\\".replace(",", "{,}"))
        M["tableRank"] = ("\\begin{tabular}{llrrrr}\n\\toprule\n & & & Teacher's own & \\multicolumn{2}{c}{Untrained student's answer} \\\\\nQuestions & Confidence & Records & answer & Qwen3.5-9B & Granite-4.1-8B \\\\\n\\midrule\n"
                          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f = RES / "thresh_stats.json"  # r = one fixed rule on the lead experiment's students (post hoc; expectations are not counted)
    if f.exists():
        S, rows = json.loads(f.read_text()), []
        pc = lambda v: "--" if v is None else f"{v:.0f}"
        ps = lambda v: "$<$1" if 0 < v < 0.5 else f"{v:.0f}"  # share passed
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            for k, K, lab in (("raw", "Raw", "own"), ("tcal", "Tcal", "t-cal."), ("scal", "Scal", "s-fit"), ("shuf", "Shuf", "shuf."),
                              ("own", "Own", "audit"), ("tcal_recal", "Recal", "fixed")):
                e = S[fam]["cells"][k]
                for t, W in ((70, "Seventy"), (80, ""), (90, "Ninety")):
                    put(f"r{fp}{K}Pass{W}", pc(e[f"t{t}"]["passed"])); put(f"r{fp}{K}Wrong{W}", pc(e[f"t{t}"]["wrong_share"]))
                a, b, d = e["t70"], e["t80"], e["t90"]
                ci = "--" if b["wrong_share"] is None else f"{b['wrong_lo']:.0f}--{b['wrong_hi']:.0f}"
                rows.append(f"{fname} & {lab} & {e['conf']:.0f} & {e['accuracy']:.0f} & {ps(a['passed'])} & {pc(a['wrong_share'])} & "
                            f"{ps(b['passed'])} & {pc(b['wrong_share'])} & {ci} & {b['wrong_per_1000']:.0f} & "
                            f"{ps(d['passed'])} & {pc(d['wrong_share'])} & {e['half_wrong']:.0f} \\\\")
        M["tableThresh"] = ("\\begin{tabular}{llrrrrrrrrrrr}\n\\toprule\n & & & & \\multicolumn{2}{c}{At least 70} & \\multicolumn{4}{c}{At least 80 (the rule)} & "
                            "\\multicolumn{2}{c}{At least 90} & \\\\\nStud. & Labels & Says & Acc. & Pass & Err. & Pass & Err. & 95\\% & Per K & "
                            "Pass & Err. & Half \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f = RES / "rep_stats.json"  # p = seeds of the shuffled control (PROTOCOL.md 03:37)
    if f.exists():
        S, rows, ph, pn = json.loads(f.read_text()), [], 0, 0
        for fam, (fp, fname) in fams:
            sd = S.get(fam, {}).get("seeds", {})
            if "predictions" not in S.get(fam, {}):
                continue
            for k in ("1", "2", "3"):
                if k not in sd:
                    continue
                e, d, h = sd[k], sd[k]["auroc_diff"], sd[k]["half_diff"]
                rows.append(f"{fname} & {k} & {f2(e['tcal']['auroc'])} & {f2(e['shuf']['auroc'])} & {sg(d['d'], 2)} & {f2(d['lo'])}--{f2(d['hi'])} & "
                            f"{e['tcal']['sel50']:.0f} & {e['shuf']['sel50']:.0f} & {sg(h['d'])} & {h['lo']:.0f}--{h['hi']:.0f} & {e['shuf']['conf']:.1f} & {sg(e['shuf']['gap'], 1)} \\\\")
            new = [sd[k] for k in ("2", "3") if k in sd]
            put(f"p{fp}AurocDiffs", " and ".join(sg(e["auroc_diff"]["d"], 2) for e in new))
            put(f"p{fp}HalfDiffs", " and ".join(sg(e["half_diff"]["d"]) for e in new))
            put(f"p{fp}ShufAurocs", " and ".join(f2(e["shuf"]["auroc"]) for e in new))
            ph += sum(S[fam]["predictions"].values()); pn += len(S[fam]["predictions"])
        if pn:
            put("pPredHeld", ph); put("pPredAll", pn)
            held += ph; n += pn
            M["tableRep"] = ("\\begin{tabular}{lrrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{AUROC} & \\multicolumn{4}{c}{More confident half, \\% right} & \\multicolumn{2}{c}{Shuffled student} \\\\\n"
                             "Stud. & Seed & T-cal. & Shuf. & Diff. & 95\\% & T-cal. & Shuf. & Diff. & 95\\% & Says & Gap \\\\\n\\midrule\n"
                             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    f = RES / "mcl_stats.json"  # n = why the fitted label went stale on multiple choice (PROTOCOL.md 03:30)
    if f.exists():
        S, nh, nn = json.loads(f.read_text()), 0, 0
        ci = lambda d: f"{sg(d['d'], 1)} ({sg(d['lo'], 1)} to {sg(d['hi'], 1)})"
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            m, c = S[fam]["matched"], S[fam]["cost"]
            put(f"n{fp}TrainDiff", ci(m["trained_minus_untrained"]["sampled"]))
            put(f"n{fp}BaseAcc", f"{m['untrained_prefix']['sampled']['accuracy']:.1f}")
            if "untrained_noprefix" in m:
                put(f"n{fp}NoPrefixFmt", f"{100 * m['untrained_noprefix']['greedy']['parsed']:.0f}")
                put(f"n{fp}PrefixDiff", ci(m["noprefix_minus_prefix"]["sampled"]))
            for k, K in (("scal", "Scal"), ("tcal", "Tcal")):
                for ns, W in (("50", "Fifty"), ("100", "Hundred"), ("250", "TwoFifty"), ("500", "FiveHundred")):
                    put(f"n{fp}{K}Cost{W}", f"{c[k][ns]['isotonic']['within5']:.0f}")
            if "lfit_forecast" in S[fam]:
                put("nShiftForecast", sg(S[fam]["lfit_forecast"]["gap_half_b"], 1))
            d = S[fam]["decomposition"]["scal"]
            put(f"n{fp}Decomp", f"{sg(d['gap'], 1)} = {sg(d['copying'], 1)} {sg(d['label_gap'], 1)} {sg(d['question_sets'], 1)} {sg(d['training'], 1)}")
            if "own" in S[fam]:
                o = S[fam]["own"]
                put(f"n{fp}OwnAccR", f"{o['sampled']['accuracy']:.0f}")
                put(f"n{fp}OwnAcc", f"{o['sampled']['accuracy']:.1f}"); put(f"n{fp}OwnConf", f"{o['sampled']['conf']:.1f}")
                put(f"n{fp}OwnLabel", f"{o['label']:.1f}"); put(f"n{fp}OwnAuroc", f2(o["sampled"]["auroc"]))
                put(f"n{fp}OwnGap", f"{sg(o['sampled']['gap'], 1)} ({sg(o['sampled']['gap_lo'], 1)} to {sg(o['sampled']['gap_hi'], 1)})")
                put(f"n{fp}OwnVsScal", ci(o["minus_scal"])); put(f"n{fp}OwnVsBase", ci(o["minus_untrained"]))
                nh += sum(S[fam]["predictions"].values()); nn += len(S[fam]["predictions"])
        if nn:
            put("nPredHeld", nh); put("nPredAll", nn)
            held += nh; n += nn
    f = RES / "nqfix_stats.json"  # q = correction on the second question set (PROTOCOL.md 16 Nov 04:0x; post hoc)
    if f.exists():
        S, rows, FX = json.loads(f.read_text()), [], ("shift", "logodds", "isotonic")
        w5, ece, trv, psa, eh, ea = [], {x: [] for x in FX}, [], [], 0, 0
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            c = S[fam]["cells"]
            eh += sum(S[fam]["expectations"].values()); ea += len(S[fam]["expectations"])
            for k, e in c.items():
                a = e["sampled"]["after"]["250"]
                w5 += [a[x]["within5"] for x in FX]; trv.append(e["sampled"]["trivia_fitted"]["gap"])
                psa += [a[x]["passed"] for x in FX]
                for x in FX:
                    ece[x].append(a[x]["ece"])
            for k, K, lab in (("tcal", "Tcal", "t-cal."), ("scal", "Scal", "s-fit"), ("shuf", "Shuf", "shuf."), ("own", "Own", "own")):
                e = c[k]["sampled"]
                b, t, a = e["before"], e["trivia_fitted"], e["after"]
                put(f"q{fp}{K}Triv", sg(t["gap"])); put(f"q{fp}{K}Within", f"{a['250']['isotonic']['within5']:.0f}")
                put(f"q{fp}{K}EceBefore", f"{b['ece']:.0f}"); put(f"q{fp}{K}EceIso", f"{a['250']['isotonic']['ece']:.0f}")
                put(f"q{fp}{K}AurocBefore", f2(b["auroc"])); put(f"q{fp}{K}AurocIso", f2(a["250"]["isotonic"]["auroc"]))
                put(f"q{fp}{K}Pass", f"{b['passed']:.0f}"); put(f"q{fp}{K}Wrong", "--" if b["wrong_share"] is None else f"{b['wrong_share']:.0f}")
                for ns, W in (("50", "Fifty"), ("100", "Hundred"), ("500", "FiveHundred")):
                    put(f"q{fp}{K}Within{W}", f"{a[ns]['isotonic']['within5']:.0f}")
                put(f"q{fp}{K}ShiftWithinFiveHundred", f"{a['500']['shift']['within5']:.0f}")
                x3 = lambda m, d=0: " & ".join(f"{a['250'][x][m]:.{d}f}" for x in FX)
                rows.append(f"{fname} & {lab} & {e['in_format']:.0f} & {sg(b['gap'])} & {sg(t['gap'])} & {x3('within5')} & "
                            f"{x3('ece')} & {f2(b['auroc'])} & {f2(a['250']['isotonic']['auroc'])} \\\\")
        if rows:
            put("qWithinMin", f"{min(w5):.0f}"); put("qWithinMax", f"{max(w5):.0f}")
            put("qTrivMin", sg(min(trv))); put("qTrivMax", sg(max(trv)))
            put("qEceShiftMax", f"{max(ece['shift']):.0f}"); put("qEceLogoddsMax", f"{max(ece['logodds']):.0f}"); put("qEceIsoMax", f"{max(ece['isotonic']):.0f}")
            put("qPassAfterMax", "0" if max(psa) == 0 else ("under 1" if max(psa) < 1 else f"{max(psa):.0f}"))
            put("qExpHeld", eh); put("qExpAll", ea)
            M["tableNqfix"] = ("\\begin{tabular}{llrrrrrrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{Gap} & \\multicolumn{3}{c}{Within 5, \\% of draws} & "
                               "\\multicolumn{3}{c}{ECE} & \\multicolumn{2}{c}{AUROC} \\\\\n"
                               "Stud. & Labels & Fmt & Before & Trivia & Shift & Log & Iso & Shift & Log & Iso & Before & Iso \\\\\n\\midrule\n"
                               + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # what the shuffle does to the records: mean label on the teacher's right and wrong answers, before and after
    lab = {k: [json.loads(l) for l in (RES / "dist" / f"ds41f__{k}" / "d_main.jsonl").open()] for k in ("tcal", "shuf")
           if (RES / "dist" / f"ds41f__{k}" / "d_main.jsonl").exists()}
    if len(lab) == 2:
        yes = lambda r: r["correct"] in (True, "True")
        mean = lambda v: sum(v) / len(v)
        for k, K in (("tcal", "Tcal"), ("shuf", "Shuf")):
            put(f"sh{K}Right", f"{mean([float(r['conf']) for r in lab[k] if yes(r)]):.0f}")
            put(f"sh{K}Wrong", f"{mean([float(r['conf']) for r in lab[k] if not yes(r)]):.0f}")
        put("shMoved", f"{sum(a['conf'] != b['conf'] for a, b in zip(lab['tcal'], lab['shuf'])):,}".replace(",", "{,}"))
        assert all(a["id"] == b["id"] and a["answer"] == b["answer"] for a, b in zip(lab["tcal"], lab["shuf"]))
    f = RES / "lvl_stats.json"  # v = level sweep (PROTOCOL.md 16 Nov 05:5x): the labels shifted to a chosen mean
    if f.exists():
        S, vh, va, dev, da_g, da_s, sl = json.loads(f.read_text()), 0, 0, [], [], [], []
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            for rd, R in (("sampled", "S"), ("greedy", "G")):
                r = S[fam][rd]
                pts = sorted(r["points"].values(), key=lambda p: p["label"])
                put(f"v{fp}Says{R}", " / ".join(f"{p['conf']:.0f}" for p in pts)); put(f"v{fp}Right{R}", " / ".join(f"{p['accuracy']:.0f}" for p in pts))
                put(f"v{fp}Slope{R}", f2(r["slopes"]["conf_on_label"])); sl.append(r["slopes"]["conf_on_label"])
                dev += [abs(p["conf"] - p["label"]) for p in pts if p["acc_vs_90"] is not None]
                (da_s if rd == "sampled" else da_g).extend(abs(p["acc_vs_90"]["d"]) for p in pts if p["acc_vs_90"] is not None)
            put("vLabels", " / ".join(f"{p['label']:.0f}" for p in sorted(S[fam]["sampled"]["points"].values(), key=lambda p: p["label"])))
            vh += sum(S[fam]["predictions"].values()); va += len(S[fam]["predictions"])
        if va:
            put("vDevMax", f"{max(dev):.1f}"); put("vAccDiffGreedyMax", f"{max(da_g):.1f}"); put("vAccDiffSampledMax", f"{max(da_s):.1f}")
            put("vSlopeMin", f2(min(sl))); put("vSlopeMax", f2(max(sl)))
            put("vPredHeld", vh); put("vPredAll", va)
            held += vh; n += va
    f = RES / "prose_stats.json"  # z = prose arm (PROTOCOL.md 16 Nov 05:4x): certainty in the wording, no number
    if f.exists():
        S = json.loads(f.read_text())
        for fam, (fp, fname) in fams:
            if fam not in S:
                continue
            c, d = S[fam]["sampled"]["cells"], S[fam]["sampled"]["diffs"]
            for k in "ACDE":
                put(f"z{fp}Says{k}", f"{c[k]['conf']:.0f}")
            put(f"z{fp}NumC", f"{c['C']['number_given']:.0f}")
            for k in "DE":
                put(f"z{fp}Def{k}", f"{c[k]['accuracy_by_form']['assertive']:.0f}"); put(f"z{fp}Prob{k}", f"{c[k]['accuracy_by_form']['probably']:.0f}")
            a = d["auroc:D-E"]
            P = S[fam]["predictions"]  # S4 is not counted where the hedged student gave a number on under half the questions (PROTOCOL.md)
            zh = sum(v and not (k.startswith("S4") and c["C"]["number_given"] < 50) for k, v in P.items())
            M["zPredHeld"] = str(int(M.get("zPredHeld", "0")) + zh); M["zPredAll"] = str(int(M.get("zPredAll", "0")) + len(P))
            put(f"z{fp}AurocDE", f"{sg(a['d'], 2)} ({sg(a['lo'], 2)} to {sg(a['hi'], 2)})")
    put("latePredHeld", held); put("latePredAll", n)


def ext_macros():
    """Extension arm (results/ext_stats.json): wording-only confidence and the second question set.
    x + student (Q, G) + Word|Nq + cell + quantity; sampled answer."""
    f = RES / "ext_stats.json"
    if not f.exists():
        return
    S = json.loads(f.read_text())
    rows, held, n = [], 0, 0
    for fam, (fp, fname) in (("qwen35", ("Q", "Qwen")), ("granite41", ("G", "Gran."))):
        c = S[fam]
        for k, K in (("tcal_w", "Tcal"), ("scal_w", "Scal")):
            e = c[k]["E"]
            put(f"x{fp}Word{K}Gap", sg(e["gap"])); put(f"x{fp}Word{K}Conf", f"{e['conf']:.0f}")
            put(f"x{fp}Word{K}Label", f"{c[k]['label']:.0f}"); put(f"x{fp}Word{K}Auroc", f2(e["auroc"]))
        for k, K in (("tcal", "Tcal"), ("scal", "Scal"), ("shuf", "Shuf"), ("own", "Own")):
            e, e2 = c[k]["E"], c[k]["E2"]
            put(f"x{fp}Nq{K}Gap", sg(e2["gap"])); put(f"x{fp}Nq{K}Conf", f"{e2['conf']:.0f}")
            put(f"x{fp}Nq{K}Acc", f"{e2['accuracy']:.0f}"); put(f"x{fp}Nq{K}Auroc", f2(e2["auroc"]))
            put(f"x{fp}Nq{K}Sel", f"{e2['sel50']:.0f}"); put(f"x{fp}Nq{K}Drop", f"{e['conf'] - e2['conf']:.0f}")
        names = {"tcal_w": "t-cal., phrases", "scal_w": "s-fit, phrases", "tcal": "t-cal.", "scal": "s-fit",
                 "shuf": "shuf.", "own": "own records"}
        for k, lab in names.items():
            e, e2 = c[k]["E"], c[k]["E2"]
            rows.append(f"{fname} & {lab} & {c[k]['label']:.1f} & {e['conf']:.1f} & {e['accuracy']:.1f} & {sg(e['gap'], 1)} & "
                        f"{f2(e['auroc'])} & {e2['conf']:.1f} & {e2['accuracy']:.1f} & {sg(e2['gap'], 1)} & {f2(e2['auroc'])} & "
                        f"{e2['sel50']:.0f} & {100 * e2['parsed']:.0f} \\\\")
        held += sum(c["predictions"].values()); n += len(c["predictions"])
    put("xPredHeld", held); put("xPredAll", n)
    M["tableExt"] = ("\\begin{tabular}{llrrrrrrrrrrr}\n\\toprule\nStud. & Labels & Label & Says & Acc. & Gap & AUC & "
                     "NQ says & NQ acc. & NQ gap & NQ AUC & NQ half & NQ fmt \\\\\n\\midrule\n"
                     + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


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
    # second grader (Claude Haiku 5.5) on cross-model students and their same-model comparisons
    hq, hl = RES / "scoring_audit_claude-haiku-5-5.json", RES / "scoring_audit_claude-haiku-5-5_llama31.json"
    if hq.exists() and hl.exists():
        H = {**{"q:" + k: v for k, v in json.loads(hq.read_text()).items()},
             **{"l:" + k: v for k, v in json.loads(hl.read_text()).items()}}
        ag = [100 * v["agreement"] for v in H.values()]
        put("haikuRuns", len(H)); put("haikuAgreeRange", f"{min(ag):.1f}--{max(ag):.1f}")
        for pre, x, W in (("q:", "xllama31", ""), ("l:", "xqwen25", "Llama")):
            cr = [v for k, v in H.items() if k.startswith(pre) and k.endswith(x)]
            sm = [v for k, v in H.items() if k.startswith(pre) and not k.endswith(x)]
            for key, K in (("gap_rule", "Rule"), ("gap_llm", "Llm")):
                off = sum(v[key] for v in cr) / len(cr) - sum(v[key] for v in sm) / len(sm)
                put(f"haiku{W}Offset{K}", sg(off, 1))
    # hosted panel
    pp = RES / "panel.json"
    if pp.exists():
        P = json.loads(pp.read_text())
        inc = {k: v for k, v in P.items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
        gaps = sorted(v["gap"] for v in inc.values())
        put("panelTested", len(P)); put("panelIncluded", len(inc))
        # Dunning-Kruger-like pattern: stated confidence barely moves with accuracy across models.
        pa = [v["accuracy"] for v in inc.values()]; pc = [v["mean_conf"] for v in inc.values()]
        ma, mc = sum(pa) / len(pa), sum(pc) / len(pc)
        put("panelAccRange", f"{min(pa):.0f} to {max(pa):.0f}"); put("panelConfRange", f"{min(pc):.0f} to {max(pc):.0f}")
        put("panelConfSlope", f2(sum((x - ma) * (y - mc) for x, y in zip(pa, pc)) / sum((x - ma) ** 2 for x in pa)))
        put("panelOver", sum(g > 0 for g in gaps)); put("panelGapMin", sg(gaps[0])); put("panelGapMax", sg(gaps[-1]))
        put("panelGapMedian", sg(gaps[len(gaps) // 2] if len(gaps) % 2 else (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2))
        put("panelConfMin", f"{min(v['mean_conf'] for v in inc.values()):.0f}")
        rows = [f"{k.split('/')[-1].replace('_', chr(92) + '_')} & {v['accuracy']:.0f} & {v['mean_conf']:.0f} & {sg(v['gap'])} & {100 * v['parsed']:.0f} \\\\"
                for k, v in sorted(inc.items(), key=lambda kv: -kv[1]["gap"])]
        M["tablePanel"] = ("\\begin{tabular}{lrrrr}\n\\toprule\nModel & Accuracy & Stated confidence & Gap & In format (\\%) \\\\\n\\midrule\n"
                           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
        put("panelExcluded", ", ".join(f"{k.split('/')[-1].replace('_', chr(92) + '_')} ({100 * v.get('parsed', 0):.0f}\\%)" for k, v in sorted(P.items()) if k not in inc) or "none")
    # second-provider check (Adaption): the panel's first turn for the models both providers list
    ap = RES / "panel_adaption.json"
    M["tableSecondProvider"] = ""
    if ap.exists():
        A = json.loads(ap.read_text())
        names = {"qwen3.8-27b": "Qwen3.8-27B", "glm-5.2": "GLM-5.2", "glm-53-flash": "GLM-5.3-Flash"}
        rows, diffs, same = [], [], []
        for k, v in A.items():
            b = v["both_parsed"]
            diffs.append(b["gap_difference"]); same.append(b["same_answer"])
            for prov, P_ in (("Nebius", "nebius"), ("Adaption", "adaption")):
                w, s = v[P_], b[P_]
                rows.append(f"{names[k] if prov == 'Nebius' else ''} & {prov} & {100 * w['parsed']:.0f} & {w['accuracy']:.0f} & "
                            f"{w['mean_conf']:.0f} & {sg(w['gap'], 1)} & {sg(s['gap'], 1)} & "
                            f"{v['n_both_parsed'] if prov == 'Nebius' else ''} \\\\")
        M["tableSecondProvider"] = ("\\begin{tabular}{llrrrrrr}\n\\toprule\nModel & Provider & In format (\\%) & Accuracy & Stated confidence & "
                                    "Gap & Gap, shared questions & Shared questions \\\\\n\\midrule\n"
                                    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
        put("spModels", len(A)); put("spMaxDiff", f"{max(abs(d) for d in diffs):.1f}")
        put("spDiffs", ", ".join(sg(d, 1) for d in diffs))
        put("spSameMin", f"{min(same):.0f}"); put("spSameMax", f"{max(same):.0f}")
        put("spReproduced", sum(abs(d) < 5 for d in diffs))
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
    distill_macros()
    ext_macros()
    late_macros()
    card_macros()
    (Path(__file__).parent / "numbers.tex").write_text(
        "% generated by paper/make_numbers.py from results/*.json -- do not edit\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
    print(f"{len(M)} macros")
    for k, v in M.items():
        if not k.startswith("table"):
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
