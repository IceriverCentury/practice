"""Prior-mean sweep with sigma_0 = 0 (the point-mass / Dirac prior limit).

For each candidate prior mean mu_0 in {0, 0.01, 0.03, 0.05, 0.07, 0.10}, we
combine it with the same data-driven likelihood used in Section 8 (real IAP
revenue, R = 100 random 50/50 splits, +5% multiplicative uplift on the
treatment arm) and a point-mass prior with sigma_0 = 0.

Mathematical note. With sigma_0 = 0 the prior precision tau_0 = 1/sigma_0^2
is infinite, so the normal-normal conjugate update from equations (14)-(16)
collapses to mu_post = mu_0, sigma_post = 0 identically. The data does not
enter the posterior at all; the only quantity that matters is how close the
chosen prior mean lies to the true uplift. The script implements this
analytically (no division by zero) but still runs the full R = 100
simulation per prior so that we can report the data-side quantities
(rho_hat, sigma_obs) for context.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "iap_revenue_2026-05-04_to_2026-05-10.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

TRUE_UPLIFT = 0.05
R = 100
PRIOR_MEANS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10]
PRIOR_SIGMA = 0.0           # point-mass prior per user request
DECISION_THRESHOLD = 0.95
RNG = np.random.default_rng(20260511)


def random_disjoint_halves(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(len(y))
    half = len(y) // 2
    return y[idx[:half]], y[idx[half:2 * half]]


def data_side_estimators(y_C: np.ndarray, y_T: np.ndarray) -> dict:
    """Run equations (1)-(9) on one split (no prior involvement)."""
    nC, nT = len(y_C), len(y_T)
    yC_mean, yT_mean = float(y_C.mean()), float(y_T.mean())
    sC = float(y_C.std(ddof=1))
    sT = float(y_T.std(ddof=1))
    seC = sC / np.sqrt(nC)
    seT = sT / np.sqrt(nT)
    delta_hat = yT_mean - yC_mean
    se_delta = float(np.sqrt(seT ** 2 + seC ** 2))
    rho_hat = delta_hat / yC_mean
    g1 = 1.0 / yC_mean
    g2 = delta_hat / (yC_mean ** 2)
    var_rho = (g1 ** 2) * (se_delta ** 2) + (g2 ** 2) * (seC ** 2)
    sigma_obs = float(np.sqrt(var_rho))
    return {
        "rho_hat": float(rho_hat),
        "sigma_obs": float(sigma_obs),
        "delta_hat": float(delta_hat),
        "se_delta": se_delta,
        "yC_raw_mean": yC_mean,
        "se_C": float(seC),
    }


def posterior_dirac(mu_0: float, est: dict) -> dict:
    """Normal-normal update in the sigma_0 -> 0 limit: posterior collapses to prior."""
    mu_post = mu_0
    sigma_post = 0.0
    # P(rho_t > 0 | data) is the indicator I[mu_post > 0].
    p_beat = 1.0 if mu_post > 0 else 0.0
    ci_low = mu_post
    ci_high = mu_post
    return {
        "mu_post": mu_post,
        "sigma_post": sigma_post,
        "p_beat_control": p_beat,
        "ci95": (ci_low, ci_high),
    }


# --- Load data ----------------------------------------------------------
df = pd.read_csv(DATA_PATH)
y_all = df["IAP_REVENUE"].to_numpy(dtype=float)

# --- Run the simulation across priors ----------------------------------
# We use the SAME R splits for every prior so the data side is held fixed
# and only the prior changes. This makes the comparison apples-to-apples.
splits_rng = np.random.default_rng(20260511)
splits = []
for _ in range(R):
    yA, yB = random_disjoint_halves(y_all, splits_rng)
    yC = yA
    yT = yB * (1.0 + TRUE_UPLIFT)
    splits.append(data_side_estimators(yC, yT))

per_prior_records = {}
summary_rows = []
for mu_0 in PRIOR_MEANS:
    runs = []
    for est in splits:
        post = posterior_dirac(mu_0, est)
        runs.append({**est, **post, "mu_0": mu_0})
    mu_post_arr = np.array([r["mu_post"] for r in runs])
    ci_lows = np.array([r["ci95"][0] for r in runs])
    ci_highs = np.array([r["ci95"][1] for r in runs])
    p_beat_arr = np.array([r["p_beat_control"] for r in runs])

    bias = float(mu_post_arr.mean() - TRUE_UPLIFT)        # constant across runs
    rmse = float(np.sqrt(np.mean((mu_post_arr - TRUE_UPLIFT) ** 2)))
    abs_err = float(abs(mu_post_arr[0] - TRUE_UPLIFT))    # all entries identical
    detected = float((p_beat_arr > DECISION_THRESHOLD).mean())
    covers = float(((ci_lows <= TRUE_UPLIFT) & (TRUE_UPLIFT <= ci_highs)).mean())

    per_prior_records[str(mu_0)] = runs
    summary_rows.append({
        "mu_0": mu_0,
        "sigma_0": PRIOR_SIGMA,
        "posterior_mean_mean": float(mu_post_arr.mean()),
        "posterior_mean_sd": float(mu_post_arr.std(ddof=1)) if mu_post_arr.std(ddof=1) > 0 else 0.0,
        "abs_error": abs_err,
        "bias": bias,
        "rmse": rmse,
        "detection_rate": detected,
        "credible_interval_coverage": covers,
    })

summary = {
    "R": R,
    "true_uplift": TRUE_UPLIFT,
    "prior_sigma": PRIOR_SIGMA,
    "decision_threshold": DECISION_THRESHOLD,
    "rows": summary_rows,
    "note": (
        "With sigma_0 = 0 the prior is a Dirac point mass at mu_0; the "
        "normal-normal posterior collapses to mu_post = mu_0, sigma_post = 0, "
        "regardless of the data. The 'best' prior is whichever mu_0 lies "
        "closest to the true uplift."
    ),
}
with open(RES_DIR / "prior_sweep.json", "w") as f:
    json.dump(summary, f, indent=2)

# --- Capture data-side variability across runs for the plot ------------
rho_hats = np.array([est["rho_hat"] for est in splits])
sigma_obs_arr = np.array([est["sigma_obs"] for est in splits])

# --- Figure 4: 1x2 panel summarizing the prior sweep -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

mu_arr = np.array(PRIOR_MEANS)
abs_err = np.abs(mu_arr - TRUE_UPLIFT)

# Left panel: posterior mean vs prior mean, with data distribution as context.
ax = axes[0]
# Show the spread of rho_hat across R runs as a violin so the reader sees the
# data side is informative but ignored.
v = ax.violinplot(
    [rho_hats * 100] * len(PRIOR_MEANS),
    positions=mu_arr * 100,
    widths=0.7,
    showmeans=False,
    showextrema=False,
    showmedians=False,
)
for body in v["bodies"]:
    body.set_facecolor("#deebf7")
    body.set_edgecolor("#9ecae1")
    body.set_alpha(0.7)

best_idx = int(np.argmin(abs_err))
colors = ["#3182bd" if i != best_idx else "#31a354" for i in range(len(PRIOR_MEANS))]
ax.scatter(mu_arr * 100, mu_arr * 100, s=110, color=colors, zorder=5, label="posterior mean (= $\\mu_0$)")
ax.axhline(TRUE_UPLIFT * 100, color="#e6550d", linestyle="--", lw=1.5, label=f"true uplift = {TRUE_UPLIFT*100:.0f}%")
for i, mu_0 in enumerate(PRIOR_MEANS):
    ax.annotate(
        f"{mu_0*100:.0f}%",
        (mu_0 * 100, mu_0 * 100),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=9,
        color=colors[i],
    )
ax.set_xlabel(r"Prior mean $\mu_0$ (%)")
ax.set_ylabel("Inferred relative uplift (%)")
ax.set_title(r"Posterior mean by prior choice ($\sigma_0 = 0$)" + "\nshaded violins: spread of $\\hat\\rho_t$ across the 100 runs (ignored by the prior)")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(-2, 12)
ax.set_ylim(-10, 17)
ax.grid(alpha=0.25)

# Right panel: absolute error from the truth, per prior.
ax = axes[1]
labels = [f"{m*100:.0f}%" for m in PRIOR_MEANS]
bars = ax.bar(
    labels,
    abs_err * 100,
    color=colors,
    edgecolor="white",
)
y_max = max(abs_err) * 100 * 1.25 + 0.5
for i, (bar, err) in enumerate(zip(bars, abs_err)):
    height = bar.get_height()
    label = f"{err*100:.1f} pp"
    if i == best_idx:
        label += "  \u2605 best"
        bar.set_color("#31a354")
        ax.scatter([bar.get_x() + bar.get_width() / 2.0], [0.18], marker="*",
                   s=320, color="#31a354", zorder=5)
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        max(height, 0.18) + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        color=("#31a354" if i == best_idx else "black"),
        fontweight=("bold" if i == best_idx else "normal"),
    )
ax.set_xlabel(r"Prior mean $\mu_0$")
ax.set_ylabel("Absolute error from truth (percentage points)")
ax.set_title("Which prior is closer to the true 5% uplift?")
ax.set_ylim(0, y_max)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_prior_sweep.png", dpi=150)
plt.close()

print(json.dumps(summary, indent=2))
