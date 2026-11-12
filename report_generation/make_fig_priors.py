# Slide version: where the dealer's cost and the client's limit are drawn, and what each side knows.
import sys
import numpy as np, matplotlib.pyplot as plt
from scipy import stats

DEALER, CLIENT, INK, MUTED = "#D55E00", "#0072B2", "#222222", "#666666"
ln = stats.lognorm(s=0.5, scale=1.0)            # distance from m, in g: median 1, 80% in [0.53, 1.9]
lo, hi = ln.ppf(0.1), ln.ppf(0.9)
x = np.linspace(0.001, 3.0, 600)

fig, ax = plt.subplots(figsize=(8, 2.25))
for sign, col in ((-1, DEALER), (1, CLIENT)):
    ax.plot(sign * x, ln.pdf(x), color=col, lw=2)
    band = (x >= lo) & (x <= hi)
    ax.fill_between(sign * x[band], ln.pdf(x[band]), color=col, alpha=0.15, lw=0)
    ax.plot([sign, sign], [0, ln.pdf(1)], color=col, lw=1, ls=":")
ax.axvline(0, color=INK, lw=1.2, ymax=0.9)
ax.text(0, 0.93, "evaluated price $m$ (public)", ha="center", va="bottom", fontsize=9, color=INK)

ax.text(-4.4, 0.5, "Dealer's cost\n$c = m - g\\cdot$LogNormal(0, 0.5)\n"
        "Dealer knows $c$.\nClient knows only this curve\n(narrowed by the band, when shown).",
        ha="center", va="center", fontsize=8.5, color=DEALER)
ax.text(4.4, 0.5, "Client's limit\n$r = m + g\\cdot$LogNormal(0, 0.5)\n"
        "Client knows $r$.\nDealer knows only this curve.",
        ha="center", va="center", fontsize=8.5, color=CLIENT)
for sign in (-1, 1):
    ax.text(sign * 4.4, 0.1, "dotted: median 1$g$\nshaded: 80% within 0.53$g$–1.9$g$",
            ha="center", va="center", fontsize=7.5, color=MUTED)
ax.set_xlim(-6.4, 6.4); ax.set_ylim(0, 1.0)
ax.set_xticks(range(-3, 4)); ax.set_xticklabels([f"{t:+d}$g$" if t else "$m$" for t in range(-3, 4)])
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("left", "top", "right")]
ax.tick_params(axis="x", colors=INK, labelsize=8, pad=2)
fig.tight_layout(pad=0.2)
for out in sys.argv[1:]:
    fig.savefig(out, dpi=200)
