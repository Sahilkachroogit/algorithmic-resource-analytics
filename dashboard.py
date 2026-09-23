"""
dashboard.py
============
Loads telemetry_raw.csv + model.pkl, generates a professional 2×2 dark-theme
dashboard and saves it as cloud_compute_dashboard.png.

Dependencies: pandas, scikit-learn, joblib, matplotlib, seaborn
Run: python dashboard.py
"""

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
plt.style.use("dark_background")
sns.set_context("talk")

PALETTE   = {"binary_exponentiation": "#00C8FF",
             "vector_reallocation":   "#FF6B6B",
             "linked_list_traversal": "#A8FF78"}
ACCENT    = "#FFD166"
GRID_CLR  = "#2E2E3A"
BG        = "#0D0D1A"
PANEL_BG  = "#12122A"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("telemetry_raw.csv",
                 dtype={"algorithm": "category",
                        "input_size": "int64",
                        "execution_time_us": "float64",
                        "memory_bytes": "float64"})

df["log_input_size"] = np.log1p(df["input_size"].to_numpy())

# ---------------------------------------------------------------------------
# 2. Load model & predict
# ---------------------------------------------------------------------------
model = joblib.load("model.pkl")

X = df[["log_input_size", "algorithm"]].copy()
preds = model.predict(X)          # shape (n, 2)

df["pred_time_us"]     = preds[:, 0]
df["pred_memory_bytes"] = preds[:, 1]
df["residual_time"]    = df["execution_time_us"] - df["pred_time_us"]

algos = df["algorithm"].cat.categories.tolist()

# ---------------------------------------------------------------------------
# 3. Cloud cost threshold
#    Mark the inflection where execution_time_us growth rate (2nd derivative
#    of log-smoothed curve) first exceeds a risk multiple vs. the minimum.
# ---------------------------------------------------------------------------
RISK_MULTIPLIER = 10   # 10× the baseline cost = "expensive" zone

# ---------------------------------------------------------------------------
# 4. Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("Cloud Compute Telemetry Dashboard",
             fontsize=22, fontweight="bold", color="white",
             y=0.97, x=0.5)

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                      left=0.07, right=0.97, top=0.91, bottom=0.07)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color="white", pad=10)
    ax.tick_params(colors="#AAAACC", labelsize=9)
    ax.xaxis.label.set_color("#AAAACC")
    ax.yaxis.label.set_color("#AAAACC")
    ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E2E4A")


# ── Plot 1 : Actual vs Predicted Execution Time ──────────────────────────────
ax = axes[0]
style_ax(ax, "① Actual vs. Predicted Execution Time")

for algo in algos:
    sub = df[df["algorithm"] == algo].sort_values("input_size")
    c   = PALETTE.get(algo, "white")
    ax.plot(sub["input_size"], sub["execution_time_us"],
            color=c, linewidth=2.0, label=f"{algo} (actual)")
    ax.plot(sub["input_size"], sub["pred_time_us"],
            color=c, linewidth=1.4, linestyle="--", alpha=0.75,
            label=f"{algo} (predicted)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Input Size  N")
ax.set_ylabel("Execution Time  (µs)")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(fontsize=7.5, framealpha=0.15, loc="upper left",
          labelcolor="white", ncol=1)


# ── Plot 2 : Memory Allocation Growth Curves ─────────────────────────────────
ax = axes[1]
style_ax(ax, "② Memory Allocation Growth Curves")

for algo in algos:
    sub = df[df["algorithm"] == algo].sort_values("input_size")
    c   = PALETTE.get(algo, "white")
    ax.fill_between(sub["input_size"], sub["memory_bytes"] / 1024,
                    alpha=0.18, color=c)
    ax.plot(sub["input_size"], sub["memory_bytes"] / 1024,
            color=c, linewidth=2.2, label=algo)

ax.set_xscale("log")
ax.set_xlabel("Input Size  N")
ax.set_ylabel("Memory Allocated  (KB)")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(fontsize=8, framealpha=0.15, labelcolor="white")


# ── Plot 3 : Residual Errors ──────────────────────────────────────────────────
ax = axes[2]
style_ax(ax, "③ ML Model Residual Errors  (Actual − Predicted)")

for algo in algos:
    sub = df[df["algorithm"] == algo].sort_values("input_size")
    c   = PALETTE.get(algo, "white")
    ax.scatter(sub["input_size"], sub["residual_time"],
               color=c, s=22, alpha=0.85, label=algo, zorder=3)
    # LOESS-style smoothed residual line via rolling mean
    smooth = sub["residual_time"].rolling(3, center=True, min_periods=1).mean()
    ax.plot(sub["input_size"], smooth, color=c, linewidth=1.4, alpha=0.6)

ax.axhline(0, color=ACCENT, linewidth=1.0, linestyle="-", alpha=0.8,
           label="Zero error")
ax.set_xscale("log")
ax.set_xlabel("Input Size  N")
ax.set_ylabel("Residual  (µs)")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(fontsize=8, framealpha=0.15, labelcolor="white")

# Annotate MAE / R²
mae_t = mean_absolute_error(df["execution_time_us"], df["pred_time_us"])
r2_t  = r2_score(df["execution_time_us"], df["pred_time_us"])
ax.text(0.98, 0.97,
        f"MAE = {mae_t:,.1f} µs\nR²   = {r2_t:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color=ACCENT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1A1A30",
                  edgecolor=ACCENT, alpha=0.85))


# ── Plot 4 : Cloud Cost Risk Threshold ───────────────────────────────────────
ax = axes[3]
style_ax(ax, "④ Cloud Cost Risk Threshold")

COST_PER_US = 1e-6   # $ per µs (normalised unit cost)

for algo in algos:
    sub  = df[df["algorithm"] == algo].sort_values("input_size")
    cost = sub["execution_time_us"] * COST_PER_US
    c    = PALETTE.get(algo, "white")

    baseline   = cost.iloc[0] if cost.iloc[0] > 0 else cost[cost > 0].iloc[0]
    risk_mask  = cost >= baseline * RISK_MULTIPLIER
    risk_start = sub["input_size"][risk_mask].min() if risk_mask.any() else None

    ax.plot(sub["input_size"], cost, color=c, linewidth=2.2, label=algo)

    if risk_start is not None:
        # Shade the expensive zone
        ax.fill_between(sub["input_size"], cost,
                        where=(sub["input_size"] >= risk_start),
                        alpha=0.20, color=c)
        ax.axvline(risk_start, color=c, linewidth=1.0,
                   linestyle=":", alpha=0.75)
        ax.text(risk_start, cost.max() * 0.88,
                f" >{int(risk_start/1e3)}K\n risk",
                color=c, fontsize=7.5, va="top")

# Global risk band annotation
ax.set_xscale("log")
ax.set_xlabel("Input Size  N")
ax.set_ylabel("Normalised Cost  ($ × 10⁻⁶)")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(fontsize=8, framealpha=0.15, labelcolor="white")

# Red risk band label
ymax = ax.get_ylim()[1]
ax.text(0.98, 0.05,
        f"Risk threshold = {RISK_MULTIPLIER}× baseline cost",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#FF6B6B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1A1A30",
                  edgecolor="#FF6B6B", alpha=0.85))

# ---------------------------------------------------------------------------
# 5. Save & exit
# ---------------------------------------------------------------------------
fig.savefig("cloud_compute_dashboard.png",
            dpi=180, facecolor=BG, bbox_inches="tight")
plt.close(fig)

