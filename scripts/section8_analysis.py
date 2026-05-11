"""Real-data worked example for Section 8 of the A/B testing methodology narrative.

Loads the IAP revenue sample for 2026-05-04 to 2026-05-10, describes the
distribution, then runs a Monte-Carlo simulation in which two disjoint 50/50
splits of the same population play the role of control and treatment, a known
+5% multiplicative uplift is injected into the treatment arm, and the
methodology from Sections 3, 4, 6, and 7 is applied to see how reliably the
injected uplift is recovered.

Outputs:
  - figures/fig1_distribution.png
  - figures/fig2_lorenz.png
  - figures/fig3_simulation.png
  - results/single_iteration.json   (numbers for the worked single iteration)
  - results/simulation_summary.json (aggregates across R iterations)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "iap_revenue_2026-05-04_to_2026-05-10.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(20260510)

# Simulation knobs (kept conservative for clarity of the writeup).
TRUE_UPLIFT = 0.05            # injected multiplicative lift on treatment arm
R = 100                       # number of Monte-Carlo iterations
PRIOR_MU = 0.0                # skeptical prior centered at no effect
PRIOR_SIGMA = 0.05            # prior SD on relative lift (5 percentage points)
DECISION_THRESHOLD = 0.95     # P(rho_t>0|data) needed to "detect"


def srm_chi_square(n_per_group: np.ndarray, p_intended: np.ndarray) -> tuple[float, float]:
    n = n_per_group.sum()
    expected = n * p_intended
    X2 = float(np.sum((n_per_group - expected) ** 2 / expected))
    df = len(n_per_group) - 1
    pval = float(1.0 - chi2.cdf(X2, df))
    return X2, pval


def group_stats(y: np.ndarray) -> dict:
    n = int(len(y))
    mean = float(y.mean())
    sd = float(y.std(ddof=1))
    se = sd / np.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": float(se)}


def analyze_split(y_C: np.ndarray, y_T: np.ndarray) -> dict:
    """Apply equations (1)-(9) and (13)-(16) for one A/B split."""
    sC = group_stats(y_C)
    sT = group_stats(y_T)

    delta_hat = sT["mean"] - sC["mean"]
    se_delta = float(np.sqrt(sT["se"] ** 2 + sC["se"] ** 2))

    y_C_raw_mean = sC["mean"]
    rho_hat = delta_hat / y_C_raw_mean

    # Delta method: Var(rho_hat) ~ (1/y_C_raw)^2 Var(delta) + (delta/y_C_raw^2)^2 Var(y_C_raw)
    g1 = 1.0 / y_C_raw_mean
    g2 = delta_hat / (y_C_raw_mean ** 2)
    var_rho = (g1 ** 2) * (se_delta ** 2) + (g2 ** 2) * (sC["se"] ** 2)
    sigma_obs = float(np.sqrt(var_rho))

    # Normal-normal conjugate update (eqs 13-16).
    tau_prior = 1.0 / (PRIOR_SIGMA ** 2)
    tau_data = 1.0 / (sigma_obs ** 2)
    tau_post = tau_prior + tau_data
    mu_post = (tau_prior * PRIOR_MU + tau_data * rho_hat) / tau_post
    sigma_post = float(np.sqrt(1.0 / tau_post))

    p_beat_control = float(1.0 - norm.cdf(-mu_post / sigma_post))
    ci95 = (mu_post - 1.96 * sigma_post, mu_post + 1.96 * sigma_post)

    return {
        "control": sC,
        "treatment": sT,
        "delta_hat": float(delta_hat),
        "se_delta": float(se_delta),
        "rho_hat": float(rho_hat),
        "sigma_obs": float(sigma_obs),
        "g1": float(g1),
        "g2": float(g2),
        "mu_post": float(mu_post),
        "sigma_post": float(sigma_post),
        "p_beat_control": p_beat_control,
        "ci95": (float(ci95[0]), float(ci95[1])),
    }


def random_disjoint_halves(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(len(y))
    half = len(y) // 2
    return y[idx[:half]], y[idx[half:2 * half]]


# ---------- 1. Load data and produce dataset-description figures ----------

df = pd.read_csv(DATA_PATH)
y_all = df["IAP_REVENUE"].to_numpy(dtype=float)
n = len(y_all)
n_zero = int((y_all == 0).sum())
n_pos = int((y_all > 0).sum())
pos = y_all[y_all > 0]

dataset_summary = {
    "n_users": n,
    "n_zero": n_zero,
    "pct_zero": n_zero / n,
    "n_paying": n_pos,
    "pct_paying": n_pos / n,
    "mean_all": float(y_all.mean()),
    "sd_all": float(y_all.std(ddof=1)),
    "median_all": float(np.median(y_all)),
    "max_all": float(y_all.max()),
    "mean_paying": float(pos.mean()),
    "median_paying": float(np.median(pos)),
    "sd_paying": float(pos.std(ddof=1)),
    "quantiles_all": {str(q): float(np.quantile(y_all, q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]},
    "quantiles_paying": {str(q): float(np.quantile(pos, q)) for q in [0.5, 0.75, 0.9, 0.95, 0.99]},
}

# Figure 1: paying-vs-non-paying composition + log-scale histogram of paying users.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].bar(
    ["Non-payers\n(Y = 0)", "Paying users\n(Y > 0)"],
    [n_zero, n_pos],
    color=["#9ecae1", "#3182bd"],
)
for i, v in enumerate([n_zero, n_pos]):
    axes[0].text(i, v, f"{v:,}\n({v/n:.1%})", ha="center", va="bottom", fontsize=9)
axes[0].set_ylabel("Number of users")
axes[0].set_title("Composition of the sample")
axes[0].set_ylim(top=n_zero * 1.18)

bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 50)
axes[1].hist(pos, bins=bins, color="#3182bd", edgecolor="white")
axes[1].set_xscale("log")
axes[1].set_xlabel("IAP revenue per paying user (USD, log scale)")
axes[1].set_ylabel("Number of paying users")
axes[1].set_title("Distribution among paying users (n = {:,})".format(n_pos))
axes[1].axvline(pos.mean(), color="#e6550d", linestyle="--", label=f"mean = ${pos.mean():.2f}")
axes[1].axvline(np.median(pos), color="#31a354", linestyle="--", label=f"median = ${np.median(pos):.2f}")
axes[1].legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_distribution.png", dpi=150)
plt.close()

# Figure 2: Lorenz curve / top-x% revenue share.
sorted_rev = np.sort(y_all)
cum_rev = np.cumsum(sorted_rev)
cum_share_rev = cum_rev / cum_rev[-1]
cum_share_users = np.arange(1, n + 1) / n
top_share_users = 1.0 - cum_share_users
top_share_rev = 1.0 - cum_share_rev
share_top_1 = float(top_share_rev[np.searchsorted(top_share_users, 0.01, side="right") - 1]) if any(top_share_users <= 0.01) else None

fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(cum_share_users * 100, cum_share_rev * 100, color="#3182bd", lw=2, label="Lorenz curve (real data)")
ax.plot([0, 100], [0, 100], color="#888", linestyle="--", lw=1, label="Perfect equality")
for q, color in [(0.99, "#e6550d"), (0.999, "#a63603")]:
    user_pct = (1 - q) * 100
    # Share of revenue from the top (1-q) of users.
    cutoff_idx = int(np.ceil(q * n)) - 1
    rev_from_top = (cum_rev[-1] - cum_rev[cutoff_idx]) / cum_rev[-1] * 100
    ax.axvline(q * 100, color=color, linestyle=":", lw=1)
    ax.text(
        q * 100 - 0.5,
        50,
        f"top {user_pct:.1f}% of users\n→ {rev_from_top:.1f}% of revenue",
        rotation=90,
        va="center",
        ha="right",
        fontsize=8,
        color=color,
    )
ax.set_xlabel("Cumulative share of users (%, lowest revenue first)")
ax.set_ylabel("Cumulative share of revenue (%)")
ax.set_title("Revenue concentration (Lorenz curve)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_lorenz.png", dpi=150)
plt.close()

# ---------- 2. Single worked iteration (traces equations 1-17) ----------

single_rng = np.random.default_rng(7)
yA, yB = random_disjoint_halves(y_all, single_rng)
# Apply uplift to "treatment" (yB).
yT_single = yB * (1.0 + TRUE_UPLIFT)
yC_single = yA

# SRM check (eq 17). By construction the split is exactly balanced.
n_per = np.array([len(yC_single), len(yT_single)])
X2, pval_srm = srm_chi_square(n_per, np.array([0.5, 0.5]))

single = analyze_split(yC_single, yT_single)
single["srm"] = {"X2": X2, "df": 1, "p": pval_srm, "n_C": int(n_per[0]), "n_T": int(n_per[1])}
single["true_uplift"] = TRUE_UPLIFT
single["prior"] = {"mu0": PRIOR_MU, "sigma0": PRIOR_SIGMA}

with open(RES_DIR / "single_iteration.json", "w") as f:
    json.dump(single, f, indent=2)

# ---------- 3. Monte-Carlo simulation (R iterations) ----------

records = []
for r in range(R):
    yA, yB = random_disjoint_halves(y_all, RNG)
    yC = yA
    yT = yB * (1.0 + TRUE_UPLIFT)
    out = analyze_split(yC, yT)
    out["iter"] = r
    records.append(out)

rho_hats = np.array([rec["rho_hat"] for rec in records])
sigma_obs_arr = np.array([rec["sigma_obs"] for rec in records])
mu_post_arr = np.array([rec["mu_post"] for rec in records])
sigma_post_arr = np.array([rec["sigma_post"] for rec in records])
p_beat_arr = np.array([rec["p_beat_control"] for rec in records])
ci_lows = np.array([rec["ci95"][0] for rec in records])
ci_highs = np.array([rec["ci95"][1] for rec in records])

# Frequentist (z-based) 95% CI on rho using delta-method SE.
freq_lo = rho_hats - 1.96 * sigma_obs_arr
freq_hi = rho_hats + 1.96 * sigma_obs_arr
freq_significant = freq_lo > 0

bayes_detected = p_beat_arr > DECISION_THRESHOLD
bayes_ci_covers_truth = (ci_lows <= TRUE_UPLIFT) & (TRUE_UPLIFT <= ci_highs)
freq_ci_covers_truth = (freq_lo <= TRUE_UPLIFT) & (TRUE_UPLIFT <= freq_hi)

sim_summary = {
    "R": R,
    "true_uplift": TRUE_UPLIFT,
    "prior": {"mu0": PRIOR_MU, "sigma0": PRIOR_SIGMA},
    "decision_threshold": DECISION_THRESHOLD,
    "rho_hat": {
        "mean": float(rho_hats.mean()),
        "sd": float(rho_hats.std(ddof=1)),
        "min": float(rho_hats.min()),
        "max": float(rho_hats.max()),
        "q025": float(np.quantile(rho_hats, 0.025)),
        "q500": float(np.quantile(rho_hats, 0.5)),
        "q975": float(np.quantile(rho_hats, 0.975)),
    },
    "sigma_obs": {
        "mean": float(sigma_obs_arr.mean()),
        "sd": float(sigma_obs_arr.std(ddof=1)),
    },
    "mu_post": {
        "mean": float(mu_post_arr.mean()),
        "sd": float(mu_post_arr.std(ddof=1)),
    },
    "sigma_post": {
        "mean": float(sigma_post_arr.mean()),
    },
    "p_beat_control": {
        "mean": float(p_beat_arr.mean()),
        "median": float(np.median(p_beat_arr)),
        "min": float(p_beat_arr.min()),
        "max": float(p_beat_arr.max()),
    },
    "detection_rate_bayes": float(bayes_detected.mean()),
    "detection_rate_freq": float(freq_significant.mean()),
    "coverage_bayes_credible_interval": float(bayes_ci_covers_truth.mean()),
    "coverage_freq_confidence_interval": float(freq_ci_covers_truth.mean()),
}

with open(RES_DIR / "simulation_summary.json", "w") as f:
    json.dump(sim_summary, f, indent=2)

# Figure 3: simulation results.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
ax.hist(rho_hats * 100, bins=20, color="#9ecae1", edgecolor="white", label=r"$\hat\rho_t$ across runs")
ax.axvline(TRUE_UPLIFT * 100, color="#e6550d", linestyle="--", lw=2, label=f"true uplift = {TRUE_UPLIFT*100:.0f}%")
ax.axvline(rho_hats.mean() * 100, color="#31a354", linestyle="--", lw=2, label=f"mean $\\hat\\rho_t$ = {rho_hats.mean()*100:.2f}%")
ax.set_xlabel(r"Estimated relative uplift $\hat\rho_t$ (%)")
ax.set_ylabel("Number of simulations")
ax.set_title(f"Distribution of $\\hat\\rho_t$ across R = {R} runs")
ax.legend(fontsize=9)

ax = axes[1]
order = np.argsort(mu_post_arr)
y_pos = np.arange(R)
ax.errorbar(
    mu_post_arr[order] * 100,
    y_pos,
    xerr=1.96 * sigma_post_arr[order] * 100,
    fmt="o",
    color="#3182bd",
    ecolor="#9ecae1",
    elinewidth=1,
    markersize=2.4,
    capsize=0,
    label="95% credible interval",
)
ax.axvline(TRUE_UPLIFT * 100, color="#e6550d", linestyle="--", lw=2, label=f"true uplift = {TRUE_UPLIFT*100:.0f}%")
ax.axvline(0, color="#888", linestyle=":", lw=1)
ax.set_xlabel("Posterior mean and 95% credible interval (%)")
ax.set_ylabel("Simulation index (sorted by posterior mean)")
ax.set_title(f"Posterior credible intervals over R = {R} runs")
ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_simulation.png", dpi=150)
plt.close()

with open(RES_DIR / "dataset_summary.json", "w") as f:
    json.dump(dataset_summary, f, indent=2)

print(json.dumps({"dataset": dataset_summary, "single": {k: v for k, v in single.items() if k in ("delta_hat","se_delta","rho_hat","sigma_obs","mu_post","sigma_post","p_beat_control","ci95","srm","control","treatment")}, "sim_summary": sim_summary}, indent=2))
