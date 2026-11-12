"""M6 figures 1-5. All values read from saved predictions / result JSONs.

Outputs: figures/figN_*.pdf|png|svg  (PNG 300 dpi for review, PDF/SVG vector)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

sys.path.insert(0, "code")
from data_pipeline import forward_labels, rank_ic, read_section  # noqa: E402
from stats_m5 import load  # noqa: E402

_STYLE = Path(__file__).resolve().parents[1] / "styles" / "sci-skills-matplotlib.mplstyle"
if _STYLE.exists():
    mpl.style.use(str(_STYLE))


C_INC = "#1F3B57"      # adapted incremental
C_RESET = "#2E8B8B"    # adapted reset
C_ZS = "#9AA0A6"       # zero-shot
C_OLS = "#D98E4A"      # aligned OLS
C_NEG = "#C0563B"      # negative / warning
OUT = "figures"


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def fig1_design():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), height_ratios=[1, 1.25])

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 3.2)
    ax.axis("off")
    ax.set_title("A  Signal, execution and accounting", loc="left")

    boxes = [
        (0.3, 1.9, 1.7, 0.7, "Signal formed\nat close $t$", C_INC),
        (2.6, 1.9, 1.7, 0.7, "Execute\nat close $t{+}1$", C_ZS),
        (4.9, 1.9, 2.1, 0.7, "Label: $r_{t+1}$ … $r_{t+h}$", "#5C6B7A"),
        (7.4, 1.9, 2.3, 0.7, "Portfolio credited\nfrom $t{+}2$ (main)", C_OLS),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor="none", alpha=0.95, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, color="white", ha="center", va="center", fontsize=7.6, zorder=3)
    for x0, x1 in [(2.0, 2.6), (4.3, 4.9), (7.0, 7.4)]:
        ax.add_patch(FancyArrowPatch((x0, 2.25), (x1, 2.25), arrowstyle="-|>", mutation_scale=9,
                                     color="0.35", lw=0.9, zorder=1))
    ax.annotate("", xy=(9.7, 1.30), xytext=(2.6, 1.30),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9))
    ax.text(6.15, 1.06,
            "credited from $t{+}2$ in the main protocol;\nfrom $t{+}1$ under the earliest-tradable convention",
            ha="center", va="center", fontsize=6.8, color="0.30", linespacing=1.3)

    ax2 = axes[1]
    ax2.set_title("B  Yearly walk-forward with purge", loc="left")
    years = list(range(2015, 2027))
    n = len(years)
    ax2.set_xlim(1999.6, 2027.3)
    ax2.set_ylim(-0.6, n + 1.25)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([str(y) for y in years])
    ax2.tick_params(axis="y", length=0, pad=2)
    ax2.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax2.set_xlabel("Calendar year", labelpad=2)
    for spine in ("left", "top", "right"):
        ax2.spines[spine].set_visible(False)

    bh = 0.30
    for i, yr in enumerate(years):
        ax2.add_patch(Rectangle((2000, i - bh / 2), yr - 21 / 252 - 2000, bh,
                                facecolor="0.85", edgecolor="none", zorder=1))
        ax2.add_patch(Rectangle((yr - 21 / 252, i - bh / 2), 21 / 252, bh,
                                facecolor=C_NEG, edgecolor="none", zorder=2))
        ax2.add_patch(Rectangle((yr, i - bh / 2), 1, bh,
                                facecolor=C_INC, edgecolor="none", zorder=2))

    handles = [
        Patch(facecolor="0.85", edgecolor="none", label="training window (expanding)"),
        Patch(facecolor=C_NEG, edgecolor="none", label="21-day purge"),
        Patch(facecolor=C_INC, edgecolor="none", label="test year"),
    ]
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3,
               frameon=False, handlelength=1.0, columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout()
    save(fig, "fig1_design")


def fig2_annual_ic():
    panel = read_section("data/100_Portfolios_10x10_Daily.csv", 19).loc["2000-01-01":]
    lab = forward_labels(panel, 5)
    paths = {
        "Adapted, incremental": ("code/tsfm_lora_yearly_5_inc_pred.npz", C_INC),
        "Adapted, reset": ("code/tsfm_lora_yearly_5_reset_pred.npz", C_RESET),
        "Zero-shot, univariate": ("code/tsfm_mv_h5_pred.npz", C_ZS),
        "Aligned OLS": ("code/baseline_ols_aligned_h5_pred.npz", C_OLS),
    }
    preds = {k: load(p, panel) for k, (p, _) in paths.items()}
    common = preds["Adapted, incremental"].index
    for k in preds:
        common = common.intersection(preds[k].index)
    common = common.sort_values()
    years = list(range(2015, 2027))
    series = {}
    for k, p in preds.items():
        ic = rank_ic(p.loc[common], lab.loc[common]).dropna()
        series[k] = ic.groupby(ic.index.year).mean().reindex(years)

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    n = len(paths)
    width = 0.2
    xs = np.arange(len(years))
    for i, (k, (_, col)) in enumerate(paths.items()):
        hatched = k == "Zero-shot, univariate"   # keeps the two mid-grey series apart in grayscale
        ax.bar(xs + (i - (n - 1) / 2) * width, series[k].values, width=width, color=col, label=k,
               edgecolor="white" if hatched else "none", linewidth=0.6, hatch="//" if hatched else None)
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Mean daily RankIC")
    ax.set_xlabel("Test year (h = 5)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=1.2, columnspacing=1.0)
    ax.set_ylim(-0.04, 0.08)
    ax.text(0.985, 0.02, "2026: January–September", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.8, color="0.40")
    fig.tight_layout()
    save(fig, "fig2_annual_ic")


def fig3_cost_frontier():
    d5 = json.load(open("code/tsfm_lora_yearly_5_inc.json", encoding="utf-8"))["schemes"]["rank"]
    d21 = json.load(open("code/tsfm_lora_yearly_21_inc.json", encoding="utf-8"))["schemes"]["rank"]
    costs = [5, 10, 20]
    y5 = [d5[f"{c}bps"]["net_sharpe"] for c in costs]
    y21 = [d21[f"{c}bps"]["net_sharpe"] for c in costs]

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(costs, y5, marker="o", color=C_INC, label="h = 5 (incremental)")
    ax.plot(costs, y21, marker="s", color=C_OLS, label="h = 21 (incremental)")
    ax.axhline(0, color="0.45", lw=0.7, ls="--")
    ax.axvline(10, color="0.82", lw=0.6)
    ax.text(10.15, 0.40, "main protocol\n(10 bps)", fontsize=7, color="0.35", va="top")

    def zero_cross(y):
        return costs[0] + (costs[1] - costs[0]) * y[0] / (y[0] - y[1])

    be5, be21 = zero_cross(y5), zero_cross(y21)
    ax.plot([be5], [0], marker="x", ms=5, mew=1.1, color=C_INC, zorder=4)
    ax.plot([be21], [0], marker="x", ms=5, mew=1.1, color=C_OLS, zorder=4)
    ax.annotate(f"break-even {be5:.1f} bps", (be5, 0), xytext=(-8, -26), textcoords="offset points",
                ha="right", fontsize=7, color=C_INC)
    ax.annotate(f"break-even {be21:.1f} bps", (be21, 0), xytext=(8, 16), textcoords="offset points",
                ha="left", fontsize=7, color=C_OLS)
    ax.annotate(f"{y5[-1]:+.3f}", (20, y5[-1]), xytext=(-6, -11), textcoords="offset points",
                ha="right", fontsize=7, color=C_INC)
    ax.annotate(f"{y21[-1]:+.3f}", (20, y21[-1]), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=7, color=C_OLS)

    ax.set_xticks(costs)
    ax.set_xlabel("One-way cost per unit turnover (bps)")
    ax.set_ylabel("Annualized net Sharpe")
    ax.set_ylim(-1.05, 0.45)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig3_cost_frontier")


def fig4_mechanism():
    g = json.load(open("code/c3_group_mode.json", encoding="utf-8"))
    p = json.load(open("code/c3_permutation.json", encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    ax.set_title("A  Parameter-matched control", loc="left")
    vals = [g["ic_pooled"], g["ic_solo"]]
    ax.bar([0, 1], vals, width=0.55, color=[C_INC, C_RESET], edgecolor="none")
    for x, v in zip([0, 1], vals):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pooled\n(group attention)", "Solo\n(masked)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("RankIC (h = 5)")
    ax.set_ylim(0, 0.0135)
    ax.text(0.5, 0.0124,
            f"prediction rank corr. = {g['pred_rank_corr_mean']:.3f}\n"
            f"$\\Delta$IC = {g['ic_diff_pooled_minus_solo']:+.4f} (NW t = {g['ic_diff_nw_t']:.2f})",
            ha="center", va="center", fontsize=7.2, color="0.25", linespacing=1.35)

    ax = axes[1]
    ax.set_title("B  Sequence-identity permutation", loc="left")
    ax.set_xlim(-0.011, 0.019)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("RankIC")
    ax.axvspan(p["null_mean"] - p["null_sd"], p["null_mean"] + p["null_sd"], color="0.90", zorder=0)
    ax.axvline(p["null_mean"], color="0.45", lw=0.9, ymin=0.05, ymax=0.95)
    ax.axvline(p["null_q99"], color="0.35", lw=1.1, ls=":", ymin=0.05, ymax=0.95)
    ax.vlines(p["observed_ic"], 0.36, 0.64, color=C_INC, lw=1.3, zorder=3)
    ax.plot([p["observed_ic"]], [0.5], marker="o", ms=7, color=C_INC, zorder=4)
    ax.annotate(f"observed = {p['observed_ic']:.4f}", (p["observed_ic"], 0.5), xytext=(8, 26),
                textcoords="offset points", fontsize=7.2, color=C_INC)
    ax.annotate(f"null 99th pct = {p['null_q99']:.4f}", (p["null_q99"], 0.5), xytext=(6, 52),
                textcoords="offset points", fontsize=7.2, color="0.30", ha="left")
    ax.text(-0.0105, 0.93, "null mean $\\pm$1 SD", fontsize=7, color="0.35", ha="left", va="top")
    ax.text(-0.0105, 0.10, f"p = {p['p_value']:.3f}; pre-registered $\\alpha$ = 0.01",
            fontsize=7.2, color="0.25", ha="left", va="bottom")
    fig.tight_layout()
    save(fig, "fig4_mechanism")


def fig5_robustness():
    c4 = json.load(open("code/c4_panels.json", encoding="utf-8"))
    c6 = json.load(open("code/c6_crisis.json", encoding="utf-8"))["h5"]["lora_inc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    ax.set_title("A  Cross-panel replication (h = 5)", loc="left")
    names = [r["panel"] for r in c4]
    xs = np.arange(len(names))
    ax.bar(xs - 0.18, [r["lora_ic"] for r in c4], width=0.34, color=C_INC, label="Adapted", edgecolor="none")
    ax.bar(xs + 0.18, [r["ols_ic"] for r in c4], width=0.34, color=C_OLS, label="Aligned OLS", edgecolor="none")
    for x, r in zip(xs, c4):
        ax.annotate(f"{r['lora_ic']:.4f}", (x - 0.18, r["lora_ic"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=C_INC)
        ax.annotate(f"{r['ols_ic']:.4f}", (x + 0.18, r["ols_ic"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=C_OLS)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Size–{r['panel']}\nnet {r['net_bps']:+.2f} bps/d" for r in c4], fontsize=7.5)
    ax.set_ylabel("RankIC")
    ax.set_ylim(0, 0.0255)
    ax.legend(loc="upper right", ncol=2, handlelength=1.1, borderaxespad=0.4)
    ax.set_xlim(-0.55, 2.55)

    ax = axes[1]
    ax.set_title("B  Crisis sub-periods, adapted model (h = 5)", loc="left")
    periods = list(c6.keys())
    vals = [c6[k]["net_bps"] for k in periods]
    cols = [C_NEG if v < 0 else C_INC for v in vals]
    ax.barh(np.arange(len(periods)), vals, height=0.55, color=cols, edgecolor="none")
    ax.set_yticks(np.arange(len(periods)))
    ax.set_yticklabels(periods)
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", lw=0.6)
    for y, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_xlabel("Net return (bps/day, 10 bps cost)")
    ax.set_xlim(-2.2, 3.6)
    fig.tight_layout()
    save(fig, "fig5_robustness")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1_design()
    fig2_annual_ic()
    fig3_cost_frontier()
    fig4_mechanism()
    fig5_robustness()
    print("wrote:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
