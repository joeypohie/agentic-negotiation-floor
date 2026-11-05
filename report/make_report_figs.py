"""Figures for the final report (CogSci two-column format).

    python report/make_report_figs.py

Run from the project folder. Reads the main episode log and the benchmark code;
writes vector PDFs (for LaTeX) and PNG previews to report/figures/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks import band_posterior                                     # noqa: E402
from disclosure_results import (load_episodes, per_scenario, bootstrap_ci, capture,  # noqa: E402
                                benchmark, counter_g, SEEDS)
from generator import make_scenario, simulate_disclosure                  # noqa: E402
from rational import price_grid, dealer_best_quote                       # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CAL = yaml.safe_load((ROOT / "config.yaml").read_text())["calibration"]
G, D = CAL["g_points"], CAL["disclosure"]
LOG = ROOT / "logs" / "main_ornith1.5-35b.jsonl"

# palette (same as figures/make_figs.py)
INK, INK2, MUTED, NEUTRAL = "#0b0b0b", "#52514e", "#8a8984", "#f0efec"
NOBAND, BAND = "#2a78d6", "#eb6834"        # slot 1 blue, slot 2 orange
DEALER, CLIENT, BANDC = "#eb6834", "#2a78d6", "#1baf7a"
COL, FULL = 3.3, 7.0                       # CogSci column and page widths, inches

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "text.parse_math": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: game and information
def fig_game():
    scn = make_scenario(1, G)
    band = simulate_disclosure(scn, G, D["K"], D["omega"], D["tau"])
    p = round(dealer_best_quote(scn.m, scn.c, G), 3)
    to_g = lambda x: (x - scn.m) / G

    fig = plt.figure(figsize=(FULL, 2.0))
    ax = fig.add_axes([0.0, 0.0, 0.43, 1.0]); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
    box = lambda x, y, w, t, ec: (ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.42), w, 0.84,
                                  boxstyle="round,pad=0.02,rounding_size=0.15", lw=1.2, ec=ec, fc="white")),
                                  ax.text(x, y, t, ha="center", va="center", fontsize=6.8))
    end = lambda x, y, t: ax.text(x, y, t, ha="center", va="center", fontsize=7,
                                  bbox=dict(boxstyle="round,pad=0.25", fc=NEUTRAL, ec="none"))
    arrow = lambda a, b: ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))
    lab = lambda x, y, t: ax.text(x, y, t, fontsize=6.5, color=INK2, style="italic", ha="center", va="center")
    ax.text(0.1, 6.1, "a  The game (one pass)", fontsize=8, weight="bold", va="top")
    box(3.6, 4.9, 4.4, "1. Dealer quotes p", DEALER)
    box(3.75, 3.15, 7.5, "2. Client accepts p, or counters q < p", CLIENT)
    box(3.6, 1.4, 6.3, "3. Dealer accepts q only if q ≥ c", DEALER)
    end(10.9, 3.15, "trade at p"); end(10.7, 2.15, "no trade"); end(10.7, 0.45, "trade at q")
    arrow((3.6, 4.48), (3.6, 3.57)); arrow((3.6, 2.73), (3.6, 1.82)); lab(4.6, 2.27, "counter")
    arrow((7.55, 3.15), (10.0, 3.15)); lab(8.8, 3.42, "accept")
    arrow((6.8, 1.6), (9.9, 2.1)); lab(8.3, 2.15, "q < c")
    arrow((6.8, 1.2), (9.9, 0.55)); lab(8.3, 0.6, "q ≥ c")

    ax = fig.add_axes([0.49, 0.2, 0.51, 0.69])
    x = np.linspace(-3.2, 3.2, 600)
    lognorm = stats.lognorm(s=0.5)
    cost = np.where(x < 0, lognorm.pdf(-x), 0)
    limit = np.where(x > 0, lognorm.pdf(x), 0)
    q = price_grid(scn.m, G)
    w, _ = band_posterior(q, scn.m, G, band.avg, band.s)
    post = np.interp(x, to_g(q), w / (q[1] - q[0]) * G)
    ax.fill_between(x, cost, color=DEALER, alpha=0.15, lw=0)
    ax.plot(x, cost, color=DEALER, lw=1, label="client's prior on dealer cost c")
    ax.fill_between(x, limit, color=CLIENT, alpha=0.12, lw=0)
    ax.plot(x, limit, color=CLIENT, lw=1, label="dealer's prior on client limit r")
    ax.plot(x, post, color=BANDC, lw=1.4, label="client's Bayesian belief about c given the band")
    ax.axvspan(to_g(band.low), to_g(band.high), ymin=0, ymax=0.06, color=BANDC, alpha=0.6, lw=0)
    for v, lab, col, ls in ((to_g(scn.c), "c", DEALER, "--"), (to_g(p), "p", DEALER, "-"),
                            (0, "m", MUTED, "-"), (to_g(scn.r), "r", CLIENT, "--")):
        ax.axvline(v, color=col, lw=0.9, ls=ls)
        ax.text(v, 2.38, lab, ha="center", va="bottom", fontsize=7.5, color=INK, style="italic")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 2.35); ax.set_yticks([])
    ax.set_xlabel("price relative to the evaluated price m, in units of g (g ≈ $3,400 on $10mm)")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=1, edgecolor="none", handlelength=1.4,
              borderaxespad=0.15, fontsize=6.6)
    ax.text(-3.2, 2.62, "b  Private values and beliefs (scenario 1); the bar marks the band's stated range",
            fontsize=8, weight="bold", va="bottom")
    save(fig, "fig1_game")


# ---------------------------------------------------------------- Figure 2: capture by arm
def fig_capture():
    recs = load_episodes(LOG)
    bench = benchmark(SEEDS, G, D)
    rU = np.mean([b["U"].client_capture for b in bench.values()])
    rB = np.mean([b["B"].client_capture for b in bench.values()])
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    groups = [("Not asked about beliefs", "U_noelicit", "B_noelicit"), ("Asked about beliefs", "U", "B")]
    for i, (name, u, b) in enumerate(groups):
        for j, (cond, col, lab) in enumerate(((u, NOBAND, "no band"), (b, BAND, "band"))):
            m, (lo, hi) = bootstrap_ci(list(per_scenario(recs, cond, capture).values()))
            xpos = i + (j - 0.5) * 0.28
            ax.errorbar(xpos, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=5, color=col, ecolor=col,
                        elinewidth=1.4, capsize=0, label=lab if i == 0 else None, zorder=3)
            ax.text(xpos + 0.06, m, f"{m:.2f}", va="center", fontsize=6.5, color=INK2)
    for val, col, lab in ((rU, NOBAND, "rational client, no band"), (rB, BAND, "rational client, band")):
        ax.axhline(val, color=col, lw=1, ls=(0, (4, 2)), zorder=1)
        ax.text(1.5, val, f"{lab} {val:.2f}", va="bottom", ha="right", fontsize=6.3, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 0.85)
    ax.set_ylabel("client's share of the surplus\n(client capture)")
    ax.legend(loc="lower left", frameon=False, ncol=2, title="LLM client, mean and 95% CI:", title_fontsize=7,
              alignment="left", borderaxespad=0.1)
    ax.grid(axis="y", color=NEUTRAL, lw=0.8); ax.set_axisbelow(True)
    save(fig, "fig2_capture")


# ---------------------------------------------------------------- Figure 3: where the client counters
def fig_counters():
    recs = load_episodes(LOG)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.75), gridspec_kw=dict(hspace=0.75))
    data = {}
    for cond in ("U_noelicit", "B_noelicit"):
        rs = [r for r in recs if r["key"]["condition"] == cond and counter_g(r, G) is not None]
        data[cond] = (np.array([counter_g(r, G) for r in rs]),
                      np.array([(r["decision"]["counter_stated"] - r["band"]["avg"]) / G for r in rs]))
    bins1, bins2 = np.arange(-2.5, 0.25, 0.05), np.arange(-2.0, 2.05, 0.05)
    for cond, col, lab in (("U_noelicit", NOBAND, "no band"), ("B_noelicit", BAND, "band")):
        a1.hist(data[cond][0], bins=bins1, histtype="step", lw=1.3, color=col, label=lab)
        a2.hist(data[cond][1], bins=bins2, histtype="step", lw=1.3, color=col, label=lab)
    S = float(np.sqrt(D["omega"] ** 2 / D["K"] + D["tau"] ** 2))       # band half-width, in g
    a1.set_ylim(0, 52); a2.set_ylim(0, 24)
    for v, t, ha in ((-1.0, "briefing's typical cost", "left"), (-1.898, "briefing's 80% edge", "right")):
        a1.axvline(v, color=INK2, lw=0.8, ls=":")
        a1.text(v + (0.03 if ha == "left" else -0.03), 50, t, fontsize=6.2, color=INK2, va="top", ha=ha)
    for v, t, y, ha in ((-S, "band range low", 23, "right"), (0.0, "band avg", 23, "left"), (S, "band range high", 17, "left")):
        a2.axvline(v, color=BANDC, lw=0.9, ls=":")
        a2.text(v + (0.03 if ha == "left" else -0.03), y, t, fontsize=6.2, color=INK2, va="top", ha=ha)
    a1.set_xlabel("counteroffer relative to the evaluated price (g)")
    a2.set_xlabel("counteroffer relative to the band's typical cost avg (g)")
    for ax, t in ((a1, "a"), (a2, "b")):
        ax.set_ylabel("counters")
        ax.text(-0.13, 1.08, t, transform=ax.transAxes, fontsize=8, weight="bold")
    a1.legend(loc="center right", frameon=False, borderaxespad=0.1)
    save(fig, "fig3_counters")


if __name__ == "__main__":
    fig_game()
    fig_capture()
    fig_counters()
    print("written to", OUT)
