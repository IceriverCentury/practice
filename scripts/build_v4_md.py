"""Build ab_testing_methodology_narrative_v4.md from v3 by appending the
prior-mean sweep as a new Section 8.6 and adding a small clarifying note
about the v3 prior."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "ab_testing_methodology_narrative_v3.md"
V4 = ROOT / "ab_testing_methodology_narrative_v4.md"

src = V3.read_text()

# 1) Add a clarifying note in 8.2 so the reader knows the v3 "skeptical
#    prior" passage will be revisited at the end of the section.
old_prior_paragraph = (
    "We adopt a *skeptical* prior $\\rho_t \\sim \\mathcal{N}(0, 0.05^2)$ — i.e.\n"
    "$\\mu_0 = 0$, $\\sigma_0 = 0.05$, $\\tau_0 = 400$ — so that the simulation\n"
    "honestly measures how much of the recovered signal comes from data\n"
    "rather than from a prior that already believes in the effect."
)
new_prior_paragraph = (
    "We adopt a *skeptical* prior $\\rho_t \\sim \\mathcal{N}(0, 0.05^2)$ — i.e.\n"
    "$\\mu_0 = 0$, $\\sigma_0 = 0.05$, $\\tau_0 = 400$ — so that the simulation\n"
    "honestly measures how much of the recovered signal comes from data\n"
    "rather than from a prior that already believes in the effect.\n"
    "**Note (v4 update).** This skeptical prior was a deliberate stress test;\n"
    "the *intended* informed prior centers the prior mean on the expected\n"
    "lift, $\\mu_0 = 0.05$, rather than the prior standard deviation.\n"
    "Section 8.6 below re-runs the experiment as a sweep across\n"
    "$\\mu_0 \\in \\{0,\\ 0.01,\\ 0.03,\\ 0.05,\\ 0.07,\\ 0.10\\}$ with\n"
    "$\\sigma_0 = 0$ (a point-mass prior) to show how the inferred uplift\n"
    "depends on this choice."
)
assert old_prior_paragraph in src, "Could not find the v3 skeptical-prior paragraph to update."
src = src.replace(old_prior_paragraph, new_prior_paragraph)

# 2) Append the new Section 8.6 right before Section 9.
section9_marker = "## 9. Tensions with current practice and directions for improvement"

NEW_SECTION_86 = r"""### 8.6 Prior sensitivity: sweeping the prior mean at $\sigma_0 = 0$

A reader of v3 pointed out that in the worked example we plugged
$0.05$ into $\sigma_0$ (the prior **standard deviation**) when the
intent of an informed prior is for $0.05$ to be the prior **mean**
$\mu_0$. To address that misinterpretation cleanly, this subsection
sweeps $\mu_0$ across six candidate values and fixes $\sigma_0 = 0$,
i.e. a *Dirac point-mass* prior, then asks which choice of $\mu_0$
recovers the injected uplift most accurately.

**Candidate priors.** With $\sigma_0 = 0$ throughout, we evaluate

$$
\mu_0 \;\in\; \{ 0.00,\ 0.01,\ 0.03,\ 0.05,\ 0.07,\ 0.10 \} ,
$$

each combined with the same $R = 100$ random 50/50 splits and the same
$+5\%$ multiplicative uplift on the treatment arm from Section 8.2.

**Methodological note.** Setting $\sigma_0 = 0$ sends the prior
precision $\tau_0 = 1/\sigma_0^2$ to infinity, so the normal-normal
conjugate update from (14)-(16) collapses to

$$
\mu_{\text{post},t} \;=\; \mu_0 ,
\qquad
\sigma_{\text{post},t} \;=\; 0 ,
\qquad
P(\rho_t > 0 \,|\, \text{data}) \;=\;
\begin{cases} 1 & \mu_0 > 0 ,\\ 0 & \mu_0 \le 0 . \end{cases}
$$

In other words, the data **does not enter the posterior at all**: the
posterior is the prior. This is the extreme case of the "rigid informed
priors" risk flagged in Section 9 — the answer depends entirely on
whether the chosen $\mu_0$ happens to be right. The simulation
machinery from Section 8.2-8.4 still runs (so the data-side estimators
$\hat\rho_t$ and $\hat\sigma_{\text{obs},t}$ are computed exactly as
before), but those numbers never make it through (15) into the
posterior.

**Results.** Figure 4 visualizes the sweep. The left panel plots the
data-side $\hat\rho_t$ distribution across the $R = 100$ runs as a
shaded violin at each $\mu_0$ (the violins are the *same shape* across
priors because the data is the same), with the resulting posterior
mean overlaid as a colored dot. With $\sigma_0 = 0$ every dot sits at
$y = \mu_0$. The right panel shows the absolute error
$|\mu_{\text{post},t} - \rho^{\star}|$ for each prior, where
$\rho^{\star} = 0.05$ is the injected truth.

![Prior-mean sweep with $\sigma_0 = 0$: posterior mean per prior choice (left) and absolute error from the true 5% uplift (right). The data-side spread of $\hat\rho_t$ is shown for context but is ignored by the point-mass prior.](figures/fig4_prior_sweep.png){ width=100% }

The numerical summary across the six priors is:

| $\mu_0$ | Posterior mean $\mu_{\text{post},t}$ | Abs. error vs. truth | $P(\rho_t > 0 \,|\, \text{data})$ | 95% CI covers $\rho^{\star}$? |
|---|---|---|---|---|
| $0.00$ | $0.0\%$ | $5.0$ pp | $0$ | no |
| $0.01$ | $1.0\%$ | $4.0$ pp | $1$ | no |
| $0.03$ | $3.0\%$ | $2.0$ pp | $1$ | no |
| $\mathbf{0.05}$ | $\mathbf{5.0\%}$ | $\mathbf{0.0\ \text{pp}}$ | $\mathbf{1}$ | **yes** |
| $0.07$ | $7.0\%$ | $2.0$ pp | $1$ | no |
| $0.10$ | $10.0\%$ | $5.0$ pp | $1$ | no |

**Which prior is best?** With $\sigma_0 = 0$, the choice of $\mu_0$ is
the *entire* answer, so "best" reduces to "closest to the true uplift."
By construction the true uplift is $\rho^{\star} = 5\%$, so
$\mu_0 = 0.05$ wins with zero error and is the only prior whose
(degenerate) 95% credible interval covers the truth. The error grows
linearly with the distance $|\mu_0 - 5\%|$: priors at $0\%$ and $10\%$
each miss by $5$ pp, $1\%$ and $9\%$-type choices miss by $\sim 4$ pp,
and $3\%$ or $7\%$ miss by $2$ pp.

**Caveat — don't read this as a recommendation to fix $\sigma_0 = 0$.**
The flat ranking above only looks like a clean ordering because the
true uplift was known in advance ($\rho^{\star} = 5\%$ was injected by
us). In a real experiment we *do not* know the answer ahead of time, so
locking in any single $\mu_0$ with $\sigma_0 = 0$ is identical to
pre-announcing the result. This is exactly the "rigid informed priors"
failure mode discussed in Section 9. The methodologically sound use of
this section is:

- If a historical $5\%$-lift expectation is well supported, set $\mu_0
  \approx 0.05$ — **but keep $\sigma_0$ comfortably positive**
  (e.g. several percentage points) so the data in (15) can move the
  posterior when reality differs from prior belief.
- If there is no strong prior, lean closer to $\mu_0 = 0$ with a
  similarly diffuse $\sigma_0$ to let the data drive the inference, at
  the cost of weaker shrinkage and slightly noisier posterior means.

The general lesson is the one the figure makes graphical: when
$\sigma_0 \to 0$, every shaded violin of data on the left panel is
discarded, and what survives is only the prior point. The pipeline is
only as accurate as that point. Letting $\sigma_0$ be positive (as in
Section 8.2-8.4) reconnects the posterior to the data and gives the
methodology its self-correcting behavior.

"""

assert section9_marker in src
src = src.replace(section9_marker, NEW_SECTION_86 + section9_marker)

V4.write_text(src)
print("v4 length lines:", src.count("\n") + 1)
