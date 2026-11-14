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


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf\n")


def fig_map(runs, name="fig1-return-map", alphas=None, title_extra=""):
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
        ax.set_xlabel("Teacher's overconfidence gap (points)")
        ax.set_xlim(-40, 50)
        ax.set_ylim(-40, 45)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Student's overconfidence gap (points)")
    axes[0].annotate("student = teacher", (-20, -20), xytext=(-12, -33), fontsize=7, color=MUTED,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED))
    axes[1].annotate("lines: share of\naudited records", (0.03, 0.97), xycoords="axes fraction",
                     va="top", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=1.5)
    save(fig, name)


def fig_gain(summary, tagfits, name="fig2-gain"):
    """Gain against audit share for both readouts; tag arm as separate marks."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
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
    ax.set_xlabel("Share of audited records in the training data (%)")
    ax.set_ylabel("Gain (student gap / teacher gap)")
    ax.set_xticks([0, 10, 25, 40, 50, 60, 75])
    ax.set_ylim(-0.1, 1.15)
    fig.tight_layout()
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
    fig.tight_layout(w_pad=1.0)
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
                alphas=config.ADDON_ALPHAS)


if __name__ == "__main__":
    main()
