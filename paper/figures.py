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
    fig.savefig(OUT / f"{name}.png", dpi=400)  # the .docx build embeds the PNG; 400 dpi stays sharp in print
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


def fig_main(name="fig8-slider", teacher="ds41f"):
    """Main-text figure, the three results as bars with their numbers on them. a, b: each student under
    three confidence labellings of the same answers: what it says against how often it is right, and
    the gap. c: how often it is right when it answers only the half of the questions it is surer about,
    trained on the calibrated labels or on the same labels shuffled; trivia and multiple choice.
    Sampled answer, seed 1."""
    need = [RES / "dist_cells.json", RES / "dist_stats.json", RES / "mc_stats.json", RES / "lvl_stats.json"]
    if not all(f.exists() for f in need):
        return
    cells, dstat, mc = (json.loads(f.read_text()) for f in need[:3])
    SAYS, SHUF = ACCENT, "#e9c9b6"
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1.4]})
    r0 = lambda v: f"{v:.0f}"

    def cell(f, tname):
        return next(r for r in cells if r["family"] == f and r["teacher"] == tname and r["readout"] == "sampled" and r["seed"] == 1)
    # a: the slider. Calibrated labels shifted to a chosen mean (level sweep) plus the calibrated cell itself.
    lv = json.loads((RES / "lvl_stats.json").read_text())
    ax = axes[0]
    ax.plot([44, 96], [44, 96], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(46, 97.5, "dashed: sure = label", fontsize=6.3, color=MUTED, ha="left", va="top")
    print(f"[{name}] a, level sweep (sampled answer, seed 1): student, label mean, says, is right")
    P = {fam: sorted(lv[fam]["sampled"]["points"].values(), key=lambda p: p["label"]) for fam in STUDENT}
    for fam, mk in zip(STUDENT, ("o", "s")):
        xs = [p["label"] for p in P[fam]]
        ax.plot(xs, [p["conf"] for p in P[fam]], color=SAYS, lw=1.3, marker=mk, ms=4.5, zorder=3)
        ax.plot(xs, [p["accuracy"] for p in P[fam]], color="#7d8791", lw=1.3, marker=mk, ms=4.5, mfc="white", mew=1.0, zorder=3)
        for p in P[fam]:
            print(f"  {STUDENT[fam]:15s} {p['label']:5.1f} {p['conf']:5.1f} {p['accuracy']:5.1f}")
    q, g = P[list(STUDENT)[0]], P[list(STUDENT)[1]]
    for k, (pq, pg) in enumerate(zip(q, g)):  # one number per level: the mean of the two students (they differ by at most 1)
        low = k == 0
        ax.annotate(f"{(pq['conf'] + pg['conf']) / 2:.0f}%", (pq["label"], pq["conf"]), xytext=(5, -5) if low else (-4, 5),
                    textcoords="offset points", ha="left" if low else "right", va="top" if low else "bottom", fontsize=7.5, color=SAYS, fontweight="bold")
    ax.text(74, 68, "how sure (%)", fontsize=7.5, color=SAYS, fontweight="bold", ha="left", va="center")
    ax.text(95, 54.5, "how correct (%)", fontsize=7.5, color=INK, fontweight="bold", ha="right", va="top")
    ax.set_xlim(44, 96); ax.set_ylim(40, 100)
    ax.set_xticks([p["label"] for p in q]); ax.set_xticklabels([f"{p['label']:.0f}" for p in q])
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlabel("confidence label in training (%)"); ax.set_ylabel("%", rotation=0, labelpad=6)
    ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title("a  The slider", fontsize=8, loc="left", fontweight="bold")

    # b: the gap under the three labellings of the lead experiment, both students
    bx = axes[1]
    sets = (("teacher's\nown", "raw"), ("teacher\ncalibrated", "tcal"), ("fitted to\nstudent", "scal"))
    bx.axhline(0, color=INK, lw=0.7, zorder=3)
    print(f"[{name}] b, overconfidence gap (sampled answer, seed 1): labels, Qwen3.5-9B, Granite-4.1-8B")
    for i, (lab, key) in enumerate(sets):
        vals = []
        for j, (fam, col) in enumerate(zip(STUDENT, (SAYS, "#d9a07f"))):
            c = cell(fam, f"{teacher}__scal_{fam}" if key == "scal" else f"{teacher}__{key}")
            gp = c["gap"]; vals.append(gp)
            bx.bar(i + (j - 0.5) * 0.4, gp, 0.38, color=col, zorder=2, label=STUDENT[fam] if i == 0 else None)
            bx.text(i + (j - 0.5) * 0.4, gp + (1.5 if gp > 0 else -1.5), f"{'+' if gp > 0 else '−'}{abs(round(gp))}", ha="center",
                    va="bottom" if gp > 0 else "top", fontsize=6.8, color=INK)
        print(f"  {lab.replace(chr(10), ' '):24s} {vals[0]:+5.1f} {vals[1]:+5.1f}")
    bx.set_xticks(range(3)); bx.set_xticklabels([x[0] for x in sets], fontsize=6.3)
    bx.set_xlim(-0.6, 2.6); bx.set_ylim(-14, 50); bx.set_yticks([0, 20, 40])
    bx.set_ylabel("percentage points", labelpad=1)
    bx.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.0, handletextpad=0.4, borderaxespad=0.1)
    bx.grid(axis="y", color=GRID, lw=0.6, zorder=0); bx.set_axisbelow(True); bx.tick_params(axis="x", length=0)
    bx.set_title("b  How sure minus how correct", fontsize=8, loc="left", fontweight="bold")

    cx = axes[2]
    print(f"[{name}] c, right on the more confident half (%), sampled answer, seed 1: questions, student, calibrated labels, shuffled, all questions (calibrated)")
    xs, ticks = [], []
    for gi, task in enumerate(("Trivia", "Multiple choice")):
        for fi, f in enumerate(STUDENT):
            x = gi * 2.5 + fi
            if gi == 0:
                w = dstat[f]["sel50:tcal-shuf"]
                a, sh, allq = w["a"], w["b"], cell(f, f"{teacher}__tcal")["accuracy"]
            else:
                a, sh = (mc[f]["cells"][k]["sampled"]["sel50"] for k in ("tcal", "shuf"))
                allq = mc[f]["cells"]["tcal"]["sampled"]["accuracy"]
            first = gi == 0 and fi == 0
            cx.bar(x - 0.2, a, 0.38, color=SAYS, zorder=2, label="calibrated labels" if first else None)
            cx.bar(x + 0.2, sh, 0.38, color=SHUF, edgecolor=ACCENT, lw=0.6, zorder=2, label="shuffled labels" if first else None)
            cx.plot([x - 0.45, x + 0.45], [allq, allq], color=INK, lw=1.1, zorder=4, label="all questions" if first else None)
            cx.text(x - 0.2, a + 1.5, r0(a), ha="center", va="bottom", fontsize=7, color=INK)
            cx.text(x + 0.2, sh + 1.5, r0(sh), ha="center", va="bottom", fontsize=7, color=INK)
            print(f"  {task:16s} {STUDENT[f]:15s} {a:5.1f} {sh:5.1f} {allq:5.1f}")
            xs.append(x); ticks.append("Qwen\n3.5-9B" if f.startswith("qwen") else "Granite\n4.1-8B")
        cx.text(gi * 2.5 + 0.5, -0.19, task, transform=cx.get_xaxis_transform(), fontsize=8, ha="center", va="top", fontweight="bold")
    cx.set_xticks(xs); cx.set_xticklabels(ticks, fontsize=6.3, linespacing=1.0)
    cx.set_xlim(-0.75, 4.25); cx.set_ylim(0, 128); cx.set_yticks([0, 20, 40, 60, 80, 100]); cx.set_ylabel("how correct (%)", labelpad=1)
    cx.legend(loc="upper left", fontsize=6.8, frameon=False, ncol=2, handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
              borderaxespad=0.1, labelspacing=0.25)
    cx.grid(axis="y", color=GRID, lw=0.6, zorder=0); cx.set_axisbelow(True); cx.tick_params(axis="x", length=0)
    cx.set_title("c  How correct on its surer half", fontsize=8, loc="left", fontweight="bold")
    fig.tight_layout(w_pad=0.9)
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
    ax.plot([], [], marker="o", ms=5, mfc="white", mec=MUTED, mew=1.0, ls="", label="trained on the same confidence labels, shuffled")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
    ax.set_xticks(xs); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_xlim(-0.75, 4.45); ax.set_ylim(0.36, 0.84); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_ylabel("AUROC of the student's confidence")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    save(fig, name)


def fig_design(name="fig7-design"):
    """Schematic of the lead experiment in four steps, each with a small drawing of what happens in it."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig, ax = plt.subplots(figsize=(6.9, 1.95))
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(0, 100)
    ax.axis("off")
    heads = ["Teacher answers", "Set the slider", "Train a student", "Test it"]
    bodies = ["A large model answers\n2,000 questions.", "Same answers; only the\nconfidence label changes.",
              "A small model learns\nfrom these records.", "1,000 new questions:\nsays against is right."]
    w, gap, y0, h = 22.6, 3.2, 3, 94
    box = lambda x, y, ww, hh, fc, ec, lw=0.0: ax.add_patch(
        FancyBboxPatch((x, y), ww, hh, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec=ec, lw=lw))
    for i in range(4):
        x = i * (w + gap)
        box(x, y0, w, h, "white", INK, 1.1)
        ax.plot([x + 3.0], [y0 + h - 10.5], marker="o", ms=10.5, mfc=ACCENT, mec=ACCENT, zorder=3)
        ax.text(x + 3.0, y0 + h - 10.9, str(i + 1), fontsize=7.5, fontweight="bold", color="white", ha="center", va="center", zorder=4)
        ax.text(x + 5.4, y0 + h - 10.5, heads[i], fontsize=8.6, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.6, y0 + h - 21, bodies[i], fontsize=7.6, color=INK, va="top", linespacing=1.3)
        box(x + 1.5, y0 + 5, w - 3, 38, "#f4f1ee", "#f4f1ee")
        cx, cy = x + 1.5, y0 + 5  # origin of the drawing area (w - 3 wide, 38 high)
        if i == 0:  # one record as the teacher wrote it
            ax.text(cx + 1.4, cy + 19, "Answer: Verdi\nConfidence: 95%", fontsize=7.8, color=INK, va="center", family="monospace", linespacing=1.4)
        elif i == 1:  # the slider with its three settings
            x0, x1, ys = cx + 2.2, cx + w - 6.2, cy + 24
            ax.plot([x0, x1], [ys, ys], color=MUTED, lw=2.2, solid_capstyle="round", zorder=2)
            for frac, lab, col in ((0.0, "55%", RAMP[5]), (0.78, "90%", ACCENT), (1.0, "98%", ACCENT)):
                xs = x0 + frac * (x1 - x0)
                ax.plot([xs], [ys], marker="o", ms=6.5, mfc=col, mec="white", mew=1.0, zorder=3)
                ax.text(xs + {"90%": -1.6, "98%": 1.2}.get(lab, 0), ys + 7.5, lab, fontsize=7.2, fontweight="bold", color=col, ha="center", va="center")
            ax.text(x0 - 1.0, cy + 9.5, "fitted to\nstudent", fontsize=6.3, color=INK, ha="left", va="center", linespacing=1.1)
            ax.text(x1 + 2.0, cy + 9.5, "calibrated / own\n(teacher)", fontsize=6.3, color=INK, ha="right", va="center", linespacing=1.1)
        elif i == 2:
            ax.text(cx + 1.4, cy + 19, "Qwen3.5-9B\nGranite-4.1-8B", fontsize=7.8, color=INK, va="center", family="monospace", linespacing=1.4)
        else:  # says against is right, and the gap between them
            bx, bw, top = cx + 2.2, w - 15.5, cy + 27
            for k, (lab, val, col) in enumerate((("says", 0.9, ACCENT), ("is right", 0.6, "#9aa3ad"))):
                yb = top - k * 13
                ax.add_patch(Rectangle((bx + 7.2, yb - 3.6), bw * val, 7.2, fc=col, ec=col, zorder=2))
                ax.text(bx + 6.4, yb, lab, fontsize=6.6, color=INK, ha="right", va="center")
            xa, xb = bx + 7.2 + bw * 0.6, bx + 7.2 + bw * 0.9
            ax.annotate("", (xb, top - 13), (xa, top - 13), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
            ax.text((xa + xb) / 2, top - 21, "gap", fontsize=7.2, fontweight="bold", color=INK, ha="center", va="center")
        if i < 3:
            ax.annotate("", (x + w + gap - 0.2, y0 + h / 2), (x + w + 0.2, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=9))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    save(fig, name)


def fig_lvl(name="fig10-levels"):
    """Level sweep: the calibrated labels shifted to a chosen mean; what the student says and how often it
    is right against the mean label of its records. Sampled answer, seed 1; diamonds: the teacher's own labels."""
    f = RES / "lvl_stats.json"
    if not f.exists():
        return
    S = json.loads(f.read_text())
    fams = [x for x in STUDENT if x in S]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.3 * len(fams), 2.6), squeeze=False)
    for pi, (ax, fam) in enumerate(zip(axes[0], fams)):
        r = S[fam]["sampled"]
        pts = sorted(r["points"].values(), key=lambda p: p["label"])
        xs = [p["label"] for p in pts]
        ax.plot([40, 100], [40, 100], color=MUTED, lw=0.8, ls="--", zorder=1, label="says = label")
        ax.plot(xs, [p["conf"] for p in pts], color=ACCENT, lw=1.4, marker="o", ms=5, zorder=3, label="says")
        ax.plot(xs, [p["accuracy"] for p in pts], color="#7d8791", lw=1.4, marker="o", ms=5, mfc="white", mew=1.1, zorder=3, label="is right")
        own = r["teachers_own"]
        ax.plot([own["label"]], [own["conf"]], marker="D", ms=4.5, mfc=ACCENT, mec=ACCENT, ls="", zorder=3)
        ax.plot([own["label"]], [own["accuracy"]], marker="D", ms=4.5, mfc="white", mec="#7d8791", mew=1.1, ls="", zorder=3)
        print(f"[{name}] {STUDENT[fam]} (sampled answer): label mean, says, is right")
        for k, p in enumerate(pts + [own]):
            low = p["conf"] < p["accuracy"]  # where the says point lies under the is-right point the two labels swap sides
            ax.annotate(f"{p['conf']:.0f}", (p["label"], p["conf"]), xytext=(0, -7) if low else (-4, 5), textcoords="offset points",
                        ha="center" if low else "right", va="top" if low else "bottom", fontsize=7.5, color=ACCENT, fontweight="bold")
            ax.annotate(f"{p['accuracy']:.0f}", (p["label"], p["accuracy"]), xytext=(0, 6) if low else (0, -7), textcoords="offset points",
                        ha="center", va="bottom" if low else "top", fontsize=7.5, color=INK)
            print(f"  {p['label']:5.1f} {p['conf']:5.1f} {p['accuracy']:5.1f}")
        ax.set_xlim(44, 102); ax.set_ylim(36, 108)
        ax.set_xticks([50, 65, 81, 90, 98]); ax.set_yticks([40, 60, 80, 100])
        ax.set_xlabel("mean confidence label in the records (%)")
        if pi == 0:
            ax.set_ylabel("percent on the test questions")
            ax.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
        ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.set_title(f"{'ab'[pi]}  {STUDENT[fam]}", fontsize=8, loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.4)
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
        fig_lvl()


if __name__ == "__main__":
    if sys.argv[1:] == ["distill"]:
        fig_distill()
        fig_dk()
        fig_main()
        fig_rank()
        fig_design()
        fig_lvl()
    else:
        main()
