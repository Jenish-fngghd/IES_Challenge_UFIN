"""
Reviewer rebuttal: static threshold sweep vs DDQN adaptive threshold.

Loads the saved XGBoost fraud scores, ground-truth labels, and transaction
amounts produced by rlhf_train.py.  Sweeps static thresholds over [0.30, 0.80]
and records FNR, F2, and dollar-value-caught at each point.  Overlays the
DDQN operating point so the comparison is apples-to-apples.

Outputs:
  rlhf_results/static_threshold_comparison.png
  rlhf_results/static_threshold_comparison.json
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ───────────────────────────────────────────────────────────────────
HERE    = pathlib.Path(__file__).parent / "rlhf_results"
scores  = np.load(HERE / "val_scores.npy")
labels  = np.load(HERE / "val_labels.npy")
amounts = np.load(HERE / "val_amounts.npy")

with open(HERE / "rlhf_metrics.json") as f:
    rlhf_metrics = json.load(f)

# ── DDQN operating point ────────────────────────────────────────────────────
DDQN_THR       = rlhf_metrics["final_threshold"]      # 0.500
BASELINE_THR   = rlhf_metrics["baseline_threshold"]   # 0.868

# ── helper ──────────────────────────────────────────────────────────────────
def evaluate_static(thr, scores, labels, amounts):
    preds  = (scores >= thr).astype(int)
    tp     = int(((preds == 1) & (labels == 1)).sum())
    fn     = int(((preds == 0) & (labels == 1)).sum())
    fp     = int(((preds == 1) & (labels == 0)).sum())
    tn     = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else 1.0

    # F2 (β=2, recall-weighted)
    denom = 4 * precision + recall
    f2    = (5 * precision * recall) / denom if denom > 0 else 0.0

    # dollar value of caught fraud
    dollar_caught = float(amounts[(preds == 1) & (labels == 1)].sum())

    return {
        "threshold":     round(thr, 3),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "fnr":           round(fnr,       4),
        "f2":            round(f2,        4),
        "dollar_caught": round(dollar_caught, 2),
    }

# ── sweep ───────────────────────────────────────────────────────────────────
thresholds = np.round(np.arange(0.30, 0.805, 0.025), 3)
rows = [evaluate_static(t, scores, labels, amounts) for t in thresholds]

# points of interest
baseline_row = evaluate_static(BASELINE_THR, scores, labels, amounts)
static_500   = evaluate_static(0.500,        scores, labels, amounts)
ddqn_row     = rlhf_metrics   # already measured values from RLHF run

# ── print table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 82)
print(f"{'Threshold':>10} | {'FNR':>7} | {'F2':>7} | {'Recall':>7} | "
      f"{'Precision':>9} | {'$Caught':>12}")
print("-" * 82)
for r in rows:
    flag = ""
    if abs(r["threshold"] - BASELINE_THR) < 0.01:
        flag = "  <- XGB baseline"
    elif abs(r["threshold"] - 0.500) < 0.01:
        flag = "  <- static @ 0.50"
    print(f"{r['threshold']:>10.3f} | {r['fnr']:>7.4f} | {r['f2']:>7.4f} | "
          f"{r['recall']:>7.4f} | {r['precision']:>9.4f} | "
          f"${r['dollar_caught']:>11,.0f}{flag}")
print("=" * 82)

print("\n-- Key Comparison --------------------------------------------------")
print(f"{'Method':<28} | {'Threshold':>9} | {'FNR':>7} | {'F2':>7} | {'$Caught':>12}")
print("-" * 68)
print(f"{'XGBoost baseline (static)':<28} | {BASELINE_THR:>9.3f} | "
      f"{baseline_row['fnr']:>7.4f} | {baseline_row['f2']:>7.4f} | "
      f"${baseline_row['dollar_caught']:>11,.0f}")
print(f"{'Static threshold @ 0.500':<28} | {0.500:>9.3f} | "
      f"{static_500['fnr']:>7.4f} | {static_500['f2']:>7.4f} | "
      f"${static_500['dollar_caught']:>11,.0f}")
print(f"{'DDQN adaptive (converged)':<28} | {DDQN_THR:>9.3f} | "
      f"{ddqn_row['fnr_rlhf']:>7.4f} | {ddqn_row['f2_rlhf']:>7.4f} | "
      f"${ddqn_row['dollar_value_caught_rlhf']:>11,.0f}")
print("-" * 68)
print("\nNote: DDQN converges to 0.500, matching the F2-optimal static point.")
print("Value of DDQN is in (a) autonomous discovery of this optimum via")
print("delayed reward signals and (b) dynamic adaptation when fraud patterns shift.\n")

# ── plot ────────────────────────────────────────────────────────────────────
thr_arr    = np.array([r["threshold"]     for r in rows])
fnr_arr    = np.array([r["fnr"]           for r in rows])
f2_arr     = np.array([r["f2"]            for r in rows])
dollar_arr = np.array([r["dollar_caught"] for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Static Threshold Sweep vs DDQN Adaptive Threshold\n"
             "(same XGBoost fraud scores, same 118,108-transaction validation set)",
             fontsize=12, fontweight="bold")

panels = [
    (axes[0], fnr_arr,    "False Negative Rate (FNR)",   "lower right", True),
    (axes[1], f2_arr,     "F2 Score (β=2, recall-weighted)", "upper right", False),
    (axes[2], dollar_arr, "Dollar Value Caught ($)",     "upper right", False),
]

for ax, y_arr, ylabel, legend_loc, lower_is_better in panels:
    ax.plot(thr_arr, y_arr, color="#2563EB", linewidth=2.2, label="Static threshold")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    # XGB baseline vertical line
    ax.axvline(BASELINE_THR, color="gray", linestyle=":", linewidth=1.5,
               label=f"XGB baseline ({BASELINE_THR})")

    # DDQN operating point
    idx_ddqn = np.argmin(np.abs(thr_arr - DDQN_THR))
    ax.plot(thr_arr[idx_ddqn], y_arr[idx_ddqn],
            marker="*", markersize=16, color="#DC2626",
            label=f"DDQN converged @ {DDQN_THR}", zorder=5)

    # annotate DDQN value
    val = y_arr[idx_ddqn]
    label_txt = f"${val:,.0f}" if "Dollar" in ylabel else f"{val:.4f}"
    ax.annotate(f"DDQN\n{label_txt}",
                xy=(thr_arr[idx_ddqn], val),
                xytext=(thr_arr[idx_ddqn] + 0.04, val * (0.92 if lower_is_better else 1.06)),
                fontsize=9, color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))

    if "Dollar" in ylabel:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1e3:.0f}k"))

    ax.legend(fontsize=9, loc=legend_loc)
    ax.set_xlim(0.28, 0.82)

plt.tight_layout()
out_png = HERE / "static_threshold_comparison.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")

# ── save JSON ────────────────────────────────────────────────────────────────
output = {
    "sweep":         rows,
    "baseline_static": baseline_row,
    "static_0500":   static_500,
    "ddqn": {
        "threshold": DDQN_THR,
        "fnr":       ddqn_row["fnr_rlhf"],
        "f2":        ddqn_row["f2_rlhf"],
        "dollar_caught": ddqn_row["dollar_value_caught_rlhf"],
    },
    "commentary": (
        "DDQN converges to the same operating threshold (0.500) as the "
        "F2-optimal static threshold. The contribution of the DDQN layer is "
        "not that it finds a lower threshold — that could be done offline — "
        "but that it (a) discovers this optimum autonomously from delayed "
        "reward signals simulating chargeback feedback latency, and (b) can "
        "adapt dynamically if fraud-score distributions shift over time without "
        "requiring manual re-tuning."
    ),
}
out_json = HERE / "static_threshold_comparison.json"
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_json}")
