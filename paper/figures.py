"""All paper figures from results/*.json. Prints every plotted number.

  python3 paper/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "loopgain"))
import analyze  # noqa: E402
import config  # noqa: E402

RES, OUT = ROOT / "results", Path(__file__).parent
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e4e4e4"
RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]  # one hue, light to dark
ACCENT = "#b3541e"
MODEL = {"qwen25": "Qwen2.5-7B", "llama31": "Llama-3.1-8B"}  # names as printed in every figure


def who(fig, teacher_fam, student_fam=None, y=0.995):
    """One line at the top of a figure: which model is the teacher and which the student."""
    student_fam = student_fam or teacher_fam
    txt = (f"Teacher: {MODEL[teacher_fam]}  ·  Student: {MODEL[student_fam]}"
           + ("  (both are fresh LoRA adapters on the same base model)" if teacher_fam == student_fam else ""))
    fig.text(0.012, y, txt, fontsize=7.5, color=INK, va="top", ha="left", fontweight="bold")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf\n")


def fig_map(runs, name="fig1-return-map", alphas=None, title_extra="", fam="qwen25"):
    """Student gap against teacher gap, one panel per readout, one line per audit share."""
    alphas = alphas or config.ALPHAS_ES
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.25), sharey=True)
    for ax, (key, label) in zip(axes, (("eval", "Greedy answer"),
                                       ("eval_sampled", "One sampled answer"))):
        ax.plot([-40, 45], [-40, 45], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=GRID, lw=0.7, zorder=0)
        ax.axvline(0, color=GRID, lw=0.7, zorder=0)
        print(f"[{name}] readout: {label}")
        ends = []
        for i, a in enumerate(alphas):
            pts = analyze.es_points(runs, a, key)
            if len(pts) < 3:
                continue
            f = analyze.fit_map(pts)
            x, y = np.array(f["x"]), np.array(f["y"])
            o = np.argsort(x)
            col = RAMP[min(i, len(RAMP) - 1)]
            ax.plot(x[o], y[o], "-o", color=col, lw=1.2, ms=3.2, zorder=3 + i)
            ends.append([y[o][-1], x[o][-1], f"{round(a * 100)}%"])
            print(f"  audit share {a:.2f}: gain {f['gain']:.2f} [{f['gain_ci'][0]:.2f}, {f['gain_ci'][1]:.2f}] "
                  f"points " + " ".join(f"({a_:+.1f},{b_:+.1f})" for a_, b_ in zip(x[o], y[o])))
        ends.sort(reverse=True)  # end labels: keep at least 4 points apart, top down
        prev = None
        for yv, xv, txt in ends:
            ly = yv if prev is None else min(yv, prev - 4.2)
            ax.annotate(txt, (xv, yv), xytext=(41.5, ly), textcoords="data", va="center",
                        fontsize=7, color=INK)
            prev = ly
        ax.set_title(label, fontsize=8.5, loc="left")
        ax.set_xlabel(f"Teacher ({MODEL[fam]}): overconfidence gap (points)")
        ax.set_xlim(-40, 50)
        ax.set_ylim(-40, 45)
        ax.set_aspect("equal")
    axes[0].set_ylabel(f"Student ({MODEL[fam]}): overconfidence gap (points)")
    axes[0].annotate("student = teacher", (-20, -20), xytext=(-12, -33), fontsize=7, color=MUTED,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    axes[1].annotate("lines: share of\naudited records", (0.03, 0.97), xycoords="axes fraction",
                     va="top", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.955))
    who(fig, fam)
    save(fig, name)


def fig_gain(summary, tagfits, name="fig2-gain"):
    """Left: gain against audit share for both readouts; tag arm as separate marks.
    Right: the weight of audit-style confidence values in the student's replies (mixture test)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.7))
    ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.axhline(0, color=GRID, lw=0.7)
    for label, col, mk in (("greedy", "#08519c", "o"), ("sampled", "#6baed6", "s")):
        xs, ys, lo, hi = [], [], [], []
        for a in config.ALPHAS_ES:
            f = summary["es_maps"].get(f"{a}:{label}")
            if f:
                xs.append(100 * a); ys.append(f["gain"]); lo.append(f["gain_ci"][0]); hi.append(f["gain_ci"][1])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="-" + mk, color=col,
                    lw=1.2, ms=3.5, capsize=1.5, elinewidth=0.7)
        ax.annotate({"greedy": "greedy answer", "sampled": "sampled answer"}[label],
                    (xs[2], ys[2]), xytext=(6, 7 if label == "greedy" else -11),
                    textcoords="offset points", fontsize=7, color=INK)
        print(f"[{name}] {label}: " + " ".join(f"{x:.0f}%:{y:.2f}[{l:.2f},{h:.2f}]" for x, y, l, h in zip(xs, ys, lo, hi)))
    xs = np.linspace(0, 75, 2)
    ax.plot(xs, 1 - xs / 100, color=MUTED, lw=0.7, ls=":")
    ax.annotate("1 − audit share", (62, 0.38), xytext=(48, 0.56), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    for a, f in tagfits.items():
        if f:
            ax.plot([100 * a], [f["gain"]], marker="D", ms=4.5, color=ACCENT, zorder=5)
            print(f"[{name}] tagged, asked as 'verified', share {a}: gain {f['gain']:.2f} {np.round(f['gain_ci'], 2).tolist()}")
    if any(tagfits.values()):
        ax.annotate("audited records tagged,\nasked under the tag", (10, 0.12), xytext=(6, 2),
                    textcoords="offset points", fontsize=6.5, color=ACCENT, va="bottom")
    ax.set_title("How much of the teacher's gap arrives", fontsize=8.5, loc="left")
    ax.set_xlabel("Share of audited records in the training data (%)")
    ax.set_ylabel("Gain (student gap / teacher gap)")
    ax.set_xticks([0, 10, 25, 40, 50, 60, 75])
    ax.set_ylim(-0.1, 1.15)

    # mixture test: weight of audit-style confidence values, mean and range over the eight teachers
    mix = summary.get("mixture", {})
    bx.axhline(0, color=GRID, lw=0.7)
    bx.axhline(1, color=GRID, lw=0.7)
    bx.plot([0, 75], [0, 0.75], color=MUTED, lw=0.7, ls=":")
    bx.axvline(50, color=GRID, lw=0.7)
    for label, col, mk, dx in (("greedy", "#08519c", "o", 0.7), ("sampled", "#6baed6", "s", -0.7)):
        xs, ys, lo, hi = [], [], [], []
        for a in config.ALPHAS_ES:
            m = mix.get(f"{a}:{label}")
            if m:
                xs.append(100 * a); ys.append(m["mean"]); lo.append(m["min"]); hi.append(m["max"])
        if not xs:
            continue
        bx.errorbar(np.array(xs) + dx, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="-" + mk, color=col,
                    lw=1.2, ms=3.5, capsize=1.5, elinewidth=0.7)
        print(f"[{name}] audit-style weight, {label}: " + " ".join(f"{x:.0f}%:{y:.2f}[{l:.2f},{h:.2f}]" for x, y, l, h in zip(xs, ys, lo, hi)))
    bx.annotate("greedy answer", (56, 0.76), xytext=(20, 0.93), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    bx.annotate("sampled answer", (59.3, 0.60), xytext=(60, 0.30), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    bx.text(78, -0.065, "dotted: weight = audit share", fontsize=6.5, color=MUTED, ha="right")
    bx.set_title("Which confidence values the student uses", fontsize=8.5, loc="left")
    bx.set_xlabel("Share of audited records in the training data (%)")
    bx.set_ylabel("Weight of audit-style values")
    bx.set_xticks([10, 25, 40, 50, 60, 75])
    bx.set_xlim(-3, 79)
    bx.set_ylim(-0.1, 1.15)
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.945))
    who(fig, "qwen25")
    save(fig, name)


def fig_chain(summary, name="fig3-chains"):
    rows = summary["chain_table"]
    arms = [(0.0, "main", "No audit"), (0.25, "main", "25% audited, sampled records"),
            (0.25, "greedy", "25% audited, greedy records")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
    for ax, (a, dec, title) in zip(axes, arms):
        ax.axhline(0, color=GRID, lw=0.7)
        for r in rows:
            if r["alpha"] != a or r["records"] != dec:
                continue
            for key, col, mk, ls in (("greedy", "#08519c", "o", "-"), ("sampled", "#6baed6", "s", "--")):
                ys = r[key]
                gx = [i for i, y in enumerate(ys) if y is not None]
                if gx:
                    ax.plot(gx, [ys[i] for i in gx], ls, marker=mk, color=col, lw=1.1, ms=3.2)
            print(f"[{name}] {title} start {r['start']}: greedy {[None if v is None else round(v, 1) for v in r['greedy']]} "
                  f"sampled {[None if v is None else round(v, 1) for v in r['sampled']]}")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Generation (0 = first teacher)")
        ax.set_xticks(range(config.CHAIN_GENS + 1))
        ax.set_ylim(-40, 40)
    axes[0].set_ylabel("Overconfidence gap (points)")
    axes[0].plot([], [], "-o", color="#08519c", ms=3.2, lw=1.1, label="greedy answer")
    axes[0].plot([], [], "--s", color="#6baed6", ms=3.2, lw=1.1, label="sampled answer")
    axes[0].legend(frameon=False, fontsize=6.5, loc="center left")
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.94))
    who(fig, "qwen25")
    save(fig, name)


def fig_cross(sq, sl, name="fig5-cross-family"):
    """Students of the other family's teachers beside students of the teachers' own family.
    Greedy answer. sq / sl: summary of the Qwen / Llama family."""
    panels = [("llama31", "qwen25", sq.get("cross", {}), sl), ("qwen25", "llama31", sl.get("cross", {}), sq)]
    if not all(p[2].get("0.0:greedy") for p in panels):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3), sharey=True)
    BLUE, GREY = "#08519c", "#8a8a8a"
    for ax, (tfam, sfam, cr, tsum) in zip(axes, panels):
        ax.plot([-40, 45], [-40, 45], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=GRID, lw=0.7, zorder=0)
        ax.axvline(0, color=GRID, lw=0.7, zorder=0)
        ends = []
        for a, ls in ((0.0, "-"), (0.75, "--")):
            ref = tsum["es_maps"].get(f"{a}:greedy")      # same teachers, student of the teachers' own family
            c = cr.get(f"{a}:greedy")
            for f, col, lab in ((ref, GREY, MODEL[tfam]), (c, BLUE, MODEL[sfam])):
                if not f:
                    continue
                x, y = np.array(f["x"]), np.array(f["y"])
                o = np.argsort(x)
                ax.plot(x[o], y[o], ls, marker="o", color=col, lw=1.2, ms=3.0, zorder=3 if col == GREY else 4)
                ends.append([y[o][-1], x[o][-1], f"{lab} student, {'no audit' if a == 0 else '75% audited'}", col])
                print(f"[{name}] teacher {MODEL[tfam]}, student {lab}, audit share {a}: gain {f['gain']:.2f} intercept {f['intercept']:+.1f} "
                      "points " + " ".join(f"({a_:+.1f},{b_:+.1f})" for a_, b_ in zip(x[o], y[o])))
        ends.sort(reverse=True)
        prev = None
        for yv, xv, txt, col in ends:
            ly = yv if prev is None else min(yv, prev - 5.0)
            ax.annotate(txt, (xv, yv), xytext=(xv + 3.5, ly), textcoords="data", va="center", fontsize=6.5, color=col)
            prev = ly
        ax.set_title(f"Teacher: {MODEL[tfam]}  ·  Student: {MODEL[sfam]}", fontsize=8, loc="left", color="#08519c")
        ax.set_xlabel(f"Teacher ({MODEL[tfam]}): overconfidence gap (points)")
        ax.set_xlim(-40, 88)
        ax.set_ylim(-48, 48)
        ax.set_xticks([-40, -20, 0, 20, 40])
    axes[0].set_ylabel("Student: overconfidence gap (points)")
    axes[0].annotate("student = teacher", (-22, -22), xytext=(-14, -38), fontsize=7, color=MUTED,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.95))
    fig.text(0.012, 0.995, "Blue: student of the other model's teachers.  Grey: student of its own model's teachers (same records).",
             fontsize=7.5, color=INK, va="top", ha="left")
    save(fig, name)


def fig_panel(name="fig4-panel"):
    p = RES / "panel.json"
    if not p.exists():
        return
    panel = {k: v for k, v in json.loads(p.read_text()).items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
    items = sorted(panel.items(), key=lambda kv: kv[1]["gap"])
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(items) + 0.7))
    ys = np.arange(len(items))
    ax.barh(ys, [v["gap"] for _, v in items], color="#4292c6", height=0.62)
    ax.axvline(0, color=INK, lw=0.6)
    for y, (k, v) in zip(ys, items):
        ax.text(max(v["gap"], 0) + 1, y, f"{v['gap']:+.0f}", va="center", fontsize=6.5, color=INK)
        print(f"[{name}] {k}: gap {v['gap']:+.1f} acc {v['accuracy']:.1f} conf {v['mean_conf']:.1f} parsed {v['parsed']:.2f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([k.split("/")[-1] for k, _ in items], fontsize=6.5)
    ax.set_xlabel("Overconfidence gap (points)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, name)


STUDENT = {"qwen35": "Qwen3.5-9B", "granite41": "Granite-4.1-8B"}


def fig_distill(name="fig0-distill", teacher="ds41f", teacher_label="DeepSeek-V4.1-Flash"):
    """Per student: accuracy and stated confidence for each confidence-label set (same answers)."""
    cells = json.loads((RES / "dist_cells.json").read_text())
    labels = json.loads((RES / "dist" / "labels.json").read_text())[teacher]["sets"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), sharey=True)
    for ax, fam in zip(axes, STUDENT):
        def cell(tname):
            c = [r for r in cells if r["family"] == fam and r["teacher"] == tname
                 and r["readout"] == "sampled" and r["seed"] == 1]
            return c[0] if c else None
        rec = labels[f"{teacher}__raw"]
        rows = [("Teacher's own records", rec["teacher_accuracy"], rec["label_mean"], MUTED)]
        for lab, tname in (("Student: teacher's confidence", f"{teacher}__raw"),
                           ("Student: calibrated for the teacher", f"{teacher}__tcal"),
                           ("Student: refitted to the student", f"{teacher}__scal_{fam}")):
            c = cell(tname)
            if c:
                rows.append((lab, c["accuracy"], c["conf"], ACCENT if "refitted" not in lab else RAMP[5]))
            if c and "calibrated" in lab and c.get("recal"):
                rows.append(("same student, corrected on held-out questions", c["accuracy"],
                             c["accuracy"] + c["recal"]["gap"], RAMP[5]))
        c = cell("base")
        if c:
            rows.append(("Control: student's own records, audited", c["accuracy"], c["conf"], MUTED))
        print(f"[{name}] {STUDENT[fam]} (sampled answer, seed 1): row, accuracy, stated confidence, gap")
        for i, (lab, acc, conf, col) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([acc, conf], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
            ax.plot([acc], [y], marker="o", ms=5, mfc="white", mec=INK, mew=0.9, zorder=3)
            ax.plot([conf], [y], marker="o", ms=5, mfc=col, mec=col, zorder=3)
            g = round(conf - acc)
            ax.annotate("0" if g == 0 else (f"{g} over" if g > 0 else f"{-g} under"), (max(acc, conf), y), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.5, color=INK)
            print(f"  {lab.strip():42s} {acc:5.1f} {conf:5.1f} {conf - acc:+5.1f}")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
        ax.set_xlim(45, 114)
        ax.set_xticks([50, 60, 70, 80, 90, 100])
        ax.set_xlabel("percent")
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Student: {STUDENT[fam]}", fontsize=8, loc="left", fontweight="bold")
    axes[0].plot([], [], marker="o", ms=5, mfc="white", mec=INK, mew=0.9, ls="", label="accuracy")
    axes[0].plot([], [], marker="o", ms=5, mfc=ACCENT, mec=ACCENT, ls="", label="stated confidence")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    fig.text(0.012, 0.995, f"Teacher: {teacher_label}  ·  at each bar: points overconfident (stated above accuracy) or underconfident",
             fontsize=7.5, color=INK, va="top", ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    save(fig, name)


def fig_dk(name="fig6-dunning-kruger"):
    """Stated confidence against accuracy: hosted panel and the distilled students (greedy answer)."""
    p = RES / "panel.json"
    if not p.exists() or not (RES / "dist_cells.json").exists():
        return
    panel = {k: v for k, v in json.loads(p.read_text()).items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
    cells = json.loads((RES / "dist_cells.json").read_text())
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.plot([30, 100], [30, 100], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(43, 40, "calibrated", rotation=45, rotation_mode="anchor", fontsize=6.5,
            color=MUTED, ha="center", va="top")
    for k, v in sorted(panel.items(), key=lambda kv: kv[1]["accuracy"]):
        ax.plot([v["accuracy"]], [v["mean_conf"]], marker="o", ms=4, mfc="white", mec=MUTED, mew=0.9, ls="", zorder=2)
        print(f"[{name}] hosted {k}: acc {v['accuracy']:.1f} conf {v['mean_conf']:.1f}")
    low = min(panel.items(), key=lambda kv: kv[1]["accuracy"])
    ax.annotate(low[0].split("/")[-1].replace("-", "-\n", 1), (low[1]["accuracy"], low[1]["mean_conf"]), xytext=(0, -7),
                textcoords="offset points", ha="center", va="top", fontsize=6.5, color=MUTED)
    ax.text(68, 92, f"{len(panel)} hosted models", fontsize=6.5, color=MUTED, ha="left", va="top")
    offs = {"qwen35": -1, "granite41": 1}
    for fam, lab in STUDENT.items():
        def cell(tname):
            return [r for r in cells if r["family"] == fam and r["teacher"] == tname
                    and r["readout"] == "greedy" and r["seed"] == 1][0]
        own, dist = cell("base"), cell("ds41f__raw")
        ax.annotate("", (dist["accuracy"], dist["conf"] - 1.5), (own["accuracy"], own["conf"] + 1.5),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9), zorder=2)
        ax.plot([own["accuracy"]], [own["conf"]], marker="o", ms=5, mfc="white", mec=ACCENT, mew=1.1, ls="", zorder=3)
        ax.plot([dist["accuracy"]], [dist["conf"]], marker="o", ms=5, mfc=ACCENT, mec=ACCENT, ls="", zorder=3)
        ax.annotate(lab, (own["accuracy"], own["conf"]), xytext=(7 * offs[fam], -1), textcoords="offset points",
                    ha="right" if offs[fam] < 0 else "left", va="center", fontsize=6.5, color=ACCENT)
        for what, c in (("own audited records", own), ("distilled from DeepSeek-V4.1-Flash", dist)):
            print(f"[{name}] {lab} {what}: acc {c['accuracy']:.1f} conf {c['conf']:.1f}")
    ax.text(57, 80, "distilled from\nthe large teacher", fontsize=6.5, color=ACCENT, ha="right", va="center")
    ax.text(63, 47, "open: the same two models trained\non their own audited records", fontsize=6.5, color=ACCENT, ha="left", va="center")
    ax.set_xlim(30, 100); ax.set_ylim(30, 102)
    ax.set_xticks(range(40, 101, 10)); ax.set_yticks(range(40, 101, 10))
    ax.set_xlabel("accuracy (%)"); ax.set_ylabel("mean stated confidence (%)")
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, name)


def fig_main(name="fig8-slider", teacher="ds41f", fam="qwen35"):
    """Main-text figure. Left: one student under the four confidence settings (same answers).
    Right: stated confidence against accuracy for the hosted panel and both students (greedy answer)."""
    p = RES / "panel.json"
    if not p.exists() or not (RES / "dist_cells.json").exists():
        return
    cells = json.loads((RES / "dist_cells.json").read_text())
    labels = json.loads((RES / "dist" / "labels.json").read_text())[teacher]["sets"]
    panel = {k: v for k, v in json.loads(p.read_text()).items() if v.get("parsed", 0) >= analyze.MIN_PARSED}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={"width_ratios": [1.5, 1]})

    def cell(f, tname, readout):
        c = [r for r in cells if r["family"] == f and r["teacher"] == tname and r["readout"] == readout and r["seed"] == 1]
        return c[0] if c else None
    rec = labels[f"{teacher}__raw"]
    rows = [("Teacher (its own records)", rec["teacher_accuracy"], rec["label_mean"], MUTED)]
    for lab, tname, col in (("Student: teacher's confidence", f"{teacher}__raw", ACCENT),
                            ("Student: teacher calibrated", f"{teacher}__tcal", ACCENT),
                            ("Student: shuffled", f"{teacher}__shuf", ACCENT),
                            ("Student: fitted to the student", f"{teacher}__scal_{fam}", RAMP[5]),
                            ("Student: own records (control)", "base", MUTED)):
        c = cell(fam, tname, "sampled")
        rows.append((lab, c["accuracy"], c["conf"], col))
    print(f"[{name}] left, {STUDENT[fam]} (sampled answer, seed 1): row, accuracy, stated confidence, gap")
    for i, (lab, acc, conf, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([acc, conf], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
        ax.plot([acc], [y], marker="o", ms=5, mfc="white", mec=INK, mew=0.9, zorder=3)
        ax.plot([conf], [y], marker="o", ms=5, mfc=col, mec=col, zorder=3)
        g = round(conf - acc)
        ax.annotate("0" if g == 0 else (f"{g} over" if g > 0 else f"{-g} under"), (max(acc, conf), y), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=7.5, color=INK)
        print(f"  {lab:34s} {acc:5.1f} {conf:5.1f} {conf - acc:+5.1f}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xlim(45, 116); ax.set_xticks([50, 60, 70, 80, 90, 100]); ax.set_xlabel("percent")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.plot([], [], marker="o", ms=5, mfc="white", mec=INK, mew=0.9, ls="", label="is right (accuracy)")
    ax.plot([], [], marker="o", ms=5, mfc=ACCENT, mec=ACCENT, ls="", label="says (stated confidence)")
    ax.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    ax.set_title(f"a  The slider: one student ({STUDENT[fam]}), same answers", fontsize=8, loc="left", fontweight="bold")

    bx.plot([30, 100], [30, 100], color=MUTED, lw=0.8, ls="--", zorder=1)
    bx.text(44, 41, "says = is right", rotation=45, rotation_mode="anchor", fontsize=6.5, color=MUTED, ha="center", va="top")
    for k, v in panel.items():
        bx.plot([v["accuracy"]], [v["mean_conf"]], marker="o", ms=4, mfc="white", mec=MUTED, mew=0.9, ls="", zorder=2)
    bx.text(66, 91, f"{len(panel)} hosted models", fontsize=6.5, color=MUTED, ha="left", va="top")
    for f in STUDENT:
        own, dist = cell(f, "base", "greedy"), cell(f, f"{teacher}__raw", "greedy")
        bx.annotate("", (dist["accuracy"], dist["conf"] - 1.5), (own["accuracy"], own["conf"] + 1.5),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9), zorder=2)
        bx.plot([own["accuracy"]], [own["conf"]], marker="o", ms=5, mfc="white", mec=ACCENT, mew=1.1, ls="", zorder=3)
        bx.plot([dist["accuracy"]], [dist["conf"]], marker="o", ms=5, mfc=ACCENT, mec=ACCENT, ls="", zorder=3)
        print(f"[{name}] right, {STUDENT[f]}: own {own['accuracy']:.1f}/{own['conf']:.1f} teacher's confidence {dist['accuracy']:.1f}/{dist['conf']:.1f}")
    bx.text(56, 80, "teacher's\nconfidence", fontsize=6.5, color=ACCENT, ha="right", va="center")
    bx.text(66, 55, "two students,\nown records", fontsize=6.5, color=ACCENT, ha="left", va="center")
    bx.set_xlim(30, 100); bx.set_ylim(30, 102)
    bx.set_xticks(range(40, 101, 20)); bx.set_yticks(range(40, 101, 20))
    bx.set_xlabel("is right (%)"); bx.set_ylabel("says (%)")
    bx.set_aspect("equal")
    bx.grid(color=GRID, lw=0.6, zorder=0); bx.set_axisbelow(True)
    bx.set_title("b  Says against is right", fontsize=8, loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    save(fig, name)


def fig_rank(name="fig9-ranking", teacher="ds41f"):
    """Result 3 and where it stops: AUROC of the student's stated confidence for its own correctness,
    trained on the calibrated teacher's confidence or on the same values shuffled across records;
    trivia and multiple-choice questions, every training seed that exists (sampled answer)."""
    mcf, repf = RES / "mc_stats.json", RES / "rep_stats.json"
    if not mcf.exists() or not (RES / "dist_cells.json").exists():
        return
    cells = json.loads((RES / "dist_cells.json").read_text())
    mc = json.loads(mcf.read_text())
    rep = json.loads(repf.read_text()) if repf.exists() else {}

    def trivia(f, lab, seed):
        if lab == "shuf" and seed > 1:
            e = rep.get(f, {}).get("seeds", {}).get(str(seed))
            return e["shuf"]["auroc"] if e else None
        c = [r for r in cells if r["family"] == f and r["teacher"] == f"{teacher}__{lab}" and r["readout"] == "sampled" and r["seed"] == seed]
        return c[0]["auroc"] if c else None

    def choice(f, lab, seed):
        c = mc.get(f, {}).get("cells", {}).get(lab if seed == 1 else f"{lab}_s{seed}")
        return c["sampled"]["auroc"] if c else None

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(4.42, 0.494, "chance", fontsize=6.5, color=MUTED, ha="right", va="top")
    xs, ticks = [], []
    print(f"[{name}] AUROC (sampled answer): questions, student, seed, calibrated teacher, shuffled")
    for gi, (task, get) in enumerate((("Trivia", trivia), ("Multiple choice", choice))):
        for fi, f in enumerate(STUDENT):
            x0 = gi * 2.3 + fi
            for seed in (1, 2, 3):
                a, sh = get(f, "tcal", seed), get(f, "shuf", seed)
                if a is None and sh is None:
                    continue
                x, big = x0 + 0.2 * (seed - 2), seed == 1
                if a is not None and sh is not None:
                    ax.plot([x, x], [sh, a], color=ACCENT, lw=1.4 if big else 0.9, zorder=2)
                if sh is not None:
                    ax.plot([x], [sh], marker="o", ms=5 if big else 3.5, mfc="white", mec=MUTED, mew=1.0, ls="", zorder=3)
                if a is not None:
                    ax.plot([x], [a], marker="o", ms=5 if big else 3.5, mfc=ACCENT, mec=ACCENT, ls="", zorder=4)
                if big:  # the seed the text reports; rounded as in the text
                    near = abs(a - sh) < 0.045  # two labels at one height: one goes above, one below
                    ax.annotate(f"{a:.2f}", (x, a), xytext=(-5, 6 if near else 0), textcoords="offset points", ha="right", va="center", fontsize=7, color=INK)
                    ax.annotate(f"{sh:.2f}", (x, sh), xytext=(-5, -10 if near else -1), textcoords="offset points", ha="right", va="center", fontsize=7, color=MUTED)
                print(f"  {task:16s} {STUDENT[f]:15s} seed {seed}: " + " ".join("  -- " if v is None else f"{v:.3f}" for v in (a, sh)))
            xs.append(x0); ticks.append(STUDENT[f])
        ax.text(gi * 2.3 + 0.5, 0.835, task, fontsize=8, color=INK, ha="center", va="top", fontweight="bold")
    ax.plot([], [], marker="o", ms=5, mfc=ACCENT, mec=ACCENT, ls="", label="trained on the calibrated teacher's confidence")
    ax.plot([], [], marker="o", ms=5, mfc="white", mec=MUTED, mew=1.0, ls="", label="trained on the same values, shuffled across records")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xticks(xs); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_xlim(-0.75, 4.45); ax.set_ylim(0.36, 0.84); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_ylabel("AUROC of the student's confidence")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, name)


def fig_design(name="fig7-design"):
    """Schematic of the lead experiment in four steps."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.9, 1.75))
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(0, 100)
    ax.axis("off")
    steps = [("1  Teacher answers", "large model,\n2,000 questions", "Answer: Verdi\nConfidence: 95%", True),
             ("2  Set the slider", "same answers,\nnew confidence", "Confidence: 95%\n -> 90% -> 55%", True),
             ("3  Train student", "small model learns\nfrom the records", "Qwen3.5-9B\nGranite-4.1-8B", True),
             ("4  Test", "1,000 new questions", "gap = says\n    - is right", True)]
    w, gap, y0, h = 22.6, 3.2, 3, 94
    for i, (head, body, mono, _) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=INK, lw=0.8))
        ax.text(x + 1.5, y0 + h - 6, head, fontsize=9, fontweight="bold", color=ACCENT, va="top")
        ax.text(x + 1.5, y0 + h - 27, body, fontsize=8, color=INK, va="top", linespacing=1.3)
        if mono:
            ax.add_patch(FancyBboxPatch((x + 1.5, y0 + 6), w - 3, 30, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc="#f3f3f3", ec="#f3f3f3", lw=0.8))
            ax.text(x + 2.5, y0 + 21, mono, fontsize=8, color=INK, va="center", family="monospace", linespacing=1.35)
        if i < 3:
            ax.annotate("", (x + w + gap - 0.3, y0 + h / 2), (x + w + 0.3, y0 + h / 2),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, name)


def main():
    runs = analyze.included(analyze.load())
    summary = json.loads((RES / "summary.json").read_text())
    fig_map(runs)
    tagfits = {a: analyze.fit_map(analyze.map_points(runs, a, tag=True, key="eval_verified"))
               for a in config.TAG_ALPHAS}
    fig_gain(summary, tagfits)
    fig_chain(summary)
    fig_panel()
    lp = RES / "runs_llama31.json"
    if lp.exists():
        fig_map(analyze.included(json.loads(lp.read_text())["runs"]), name="figA-return-map-llama",
                alphas=config.ADDON_ALPHAS, fam="llama31")
        ls = RES / "summary_llama31.json"
        if ls.exists():
            fig_cross(summary, json.loads(ls.read_text()))
    if (RES / "dist_cells.json").exists():
        fig_distill()
        fig_dk()
        fig_main()
        fig_rank()
        fig_design()


if __name__ == "__main__":
    if sys.argv[1:] == ["distill"]:
        fig_distill()
        fig_dk()
        fig_main()
        fig_rank()
        fig_design()
    else:
        main()
