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


if __name__ == "__main__":
    main()
