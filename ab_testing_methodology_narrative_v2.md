# Statistical Methodology for In-House A/B Testing

*Narrative Methods Description (Game Live-Ops Context)*

## 1. Purpose and scope

A core part of live operations at a game studio is designing and running
controlled experiments — the simplest form being A/B tests — to optimize
online products. With dozens of experiments run each year, a coherent,
auditable statistical story matters for both product decisions and long-term
learning. This note describes how user-level data flow into summaries, how
variance reduction and relative effects are defined, how Bayesian inference
sits on top of those summaries, and where frequentist diagnostics and known
limitations fit. The goal is a single narrative rather than a list of
disconnected formulas.

## 2. Notation

Symbols are introduced once here and reused throughout. We distinguish
*population* quantities (unobserved, no hat) from *sample-based estimators*
(hat or bar).

| Symbol | Meaning |
|---|---|
| $g$ | Assignment group index; $g \in \{C, 1, 2, \dots, K-1\}$, with $C$ the control group and $t \in \{1, \dots, K-1\}$ the $t$-th treatment variant. |
| $K$ | Number of groups (control + treatment variants). |
| $i \mapsto g$ | "User $i$ is assigned to group $g$" (reads as "$i$ maps to $g$"). |
| $Y_{i,g}$ | User-level outcome (KPI) for user $i$ in group $g$, after KPI-specific filters. |
| $\mathbf{x}_i$ | Vector of pre-experiment covariates for user $i$ (e.g. pre-period spend, sessions, tenure), with missingness indicators appended. |
| $n_g$ | Number of users in group $g$ with a non-null outcome. |
| $\mu_g$ | Population mean of $Y$ in group $g$, i.e. $\mathbb{E}[Y_{i,g}]$. Estimated by $\hat\mu_g$. |
| $\bar Y_g$ | Sample mean of $Y$ in group $g$ (an estimator of $\mu_g$). |
| $s_g$ | Sample standard deviation of $Y$ in group $g$. |
| $\widehat{\mathrm{SE}}(\bar Y_g)$ | Estimated standard error of $\bar Y_g$. |
| $\Delta_t$ | Population absolute treatment effect of variant $t$ vs. control: $\Delta_t = \mu_t - \mu_C$. |
| $\hat\Delta_t$ | Estimator of $\Delta_t$: $\hat\Delta_t = \bar Y_t - \bar Y_C$. |
| $\mu_C^{\text{raw}}$ | Control-group population mean on the *raw* (non-CUPED) KPI. Estimated by $\bar Y_C^{\text{raw}}$. |
| $\rho_t$ | Population relative treatment effect, $\rho_t = \Delta_t / \mu_C^{\text{raw}}$. This is also the latent parameter updated in the Bayesian step (Section 6); no separate symbol is used. |
| $\hat\rho_t$ | Sample estimator of $\rho_t$: $\hat\rho_t = \hat\Delta_t / \bar Y_C^{\text{raw}}$. Treated in the Bayesian step as a noisy measurement of $\rho_t$. |
| $\hat\sigma_{\text{obs},t}$ | Estimated standard error of $\hat\rho_t$ from the delta method. |
| $\mu_0,\ \sigma_0^2$ | Prior mean and variance for $\rho_t$. |
| $\mu_{\text{post},t},\ \sigma_{\text{post},t}^2$ | Posterior mean and variance of $\rho_t$ given data. |
| $\tau_0,\ \tau_{\text{data},t},\ \tau_{\text{post},t}$ | Precisions: $\tau_0 = 1/\sigma_0^2$, $\tau_{\text{data},t} = 1/\hat\sigma_{\text{obs},t}^2$, $\tau_{\text{post},t} = \tau_0 + \tau_{\text{data},t}$. |
| $\Phi(\cdot)$ | CDF of the standard normal $\mathcal{N}(0,1)$ distribution. |

## 3. From randomization to group-level summaries

### 3.1 Variants, control, and estimands

Users are assigned to a control group and one or more treatment variants
indexed by $t \in \{1, \dots, K-1\}$ (non-control test groups). For a fixed
key performance indicator (KPI), the outcome $Y_{i,g}$ is observed for each
user in the analysis window. Under standard assumptions — stable unit
treatment values and no interference between users — contrasts $\mu_t - \mu_C$
identify the average treatment effect of variant $t$ on the user-level
outcome. The estimand of primary interest is the *relative* version
(for $\mu_C^{\text{raw}} \neq 0$) defined in Section 4.2.

### 3.2 Means and standard errors at the group level

After preprocessing, analysis works with aggregates by assignment group.
Let $\mathcal{I}_g = \{ i : i \mapsto g \}$ be the index set of users
assigned to group $g$. Then

$$
\hat\mu_g \;=\; \bar Y_g \;=\; \frac{1}{n_g} \sum_{i \in \mathcal{I}_g} Y_i ,
\tag{1}
$$

$$
s_g^2 \;=\; \frac{1}{n_g - 1} \sum_{i \in \mathcal{I}_g} \bigl( Y_i - \bar Y_g \bigr)^2 ,
\tag{2}
$$

$$
\widehat{\mathrm{SE}}\bigl(\bar Y_g\bigr) \;=\; \frac{s_g}{\sqrt{n_g}} .
\tag{3}
$$

Each assignment group contributes a pair $\bigl(\bar Y_g,\ \widehat{\mathrm{SE}}(\bar Y_g)\bigr)$
that summarizes the evidence for that variant or the control.

## 4. Absolute and relative treatment effects

### 4.1 Absolute uplift

For each treatment variant $t \in \{1, \dots, K-1\}$, the absolute uplift
estimator is

$$
\hat\Delta_t \;=\; \bar Y_t - \bar Y_C .
\tag{4}
$$

Randomization makes $\bar Y_t$ and $\bar Y_C$ approximately independent
(conditional on the realized sample sizes $n_t, n_C$), so

$$
\widehat{\mathrm{SE}}\bigl(\hat\Delta_t\bigr)
\;=\; \sqrt{ \widehat{\mathrm{SE}}(\bar Y_t)^2 + \widehat{\mathrm{SE}}(\bar Y_C)^2 } .
\tag{5}
$$

Equations (4)–(5) support standard large-sample normal reporting for
$\hat\Delta_t$ and feed the relative-uplift step.

### 4.2 Relative uplift and a consistent denominator

Stakeholders typically interpret results as percentage change relative to
control. Two conventions matter here. First, the denominator is *always*
the control mean on the **raw** (non-CUPED) KPI, $\bar Y_C^{\text{raw}}$,
even when the numerator $\hat\Delta_t$ uses a CUPED-adjusted outcome
(Section 5). Second, only the numerator carries the variance-reduction
benefit; the business denominator is fixed to the raw baseline so that
"% lift" is interpretable on a single scale across experiments. The
relative uplift estimator is

$$
\hat\rho_t \;=\; \frac{\hat\Delta_t}{\bar Y_C^{\text{raw}}} .
\tag{6}
$$

### 4.3 Uncertainty for ratios via the delta method

The relative uplift is a nonlinear function of two estimated means.
Write $\hat\rho_t = f\bigl(\hat\Delta_t, \bar Y_C^{\text{raw}}\bigr)$ with
$f(a, b) = a/b$. A first-order Taylor expansion of $f$ around the
population values $(\Delta_t, \mu_C^{\text{raw}})$ gives

$$
\hat\rho_t \;\approx\; \frac{\Delta_t}{\mu_C^{\text{raw}}}
\;+\; \frac{1}{\mu_C^{\text{raw}}}\bigl(\hat\Delta_t - \Delta_t\bigr)
\;-\; \frac{\Delta_t}{\bigl(\mu_C^{\text{raw}}\bigr)^2}\bigl(\bar Y_C^{\text{raw}} - \mu_C^{\text{raw}}\bigr) ,
\tag{7}
$$

so the coefficients in the delta-method variance are exactly the two
partial derivatives of the ratio function. Assuming $\hat\Delta_t$ and
$\bar Y_C^{\text{raw}}$ are approximately independent (a standard working
assumption under randomization),

$$
\widehat{\mathrm{Var}}\bigl(\hat\rho_t\bigr)
\;\approx\;
\left( \frac{1}{\bar Y_C^{\text{raw}}} \right)^{\!2}
\widehat{\mathrm{Var}}\bigl(\hat\Delta_t\bigr)
\;+\;
\left( \frac{\hat\Delta_t}{\bigl(\bar Y_C^{\text{raw}}\bigr)^2} \right)^{\!2}
\widehat{\mathrm{Var}}\bigl(\bar Y_C^{\text{raw}}\bigr) ,
\tag{8}
$$

where $\widehat{\mathrm{Var}}(\hat\Delta_t) = \widehat{\mathrm{SE}}(\hat\Delta_t)^2$
and $\widehat{\mathrm{Var}}(\bar Y_C^{\text{raw}}) = \widehat{\mathrm{SE}}(\bar Y_C^{\text{raw}})^2$.
We report

$$
\hat\sigma_{\text{obs},t} \;=\; \sqrt{\widehat{\mathrm{Var}}\bigl(\hat\rho_t\bigr)} .
\tag{9}
$$

The same gradient logic extends to KPIs defined as ratios of two per-user
metrics and to compound relative uplifts built from numerator and
denominator KPIs, yielding a single scalar $\hat\sigma_{\text{obs},t}$ in
each case.

## 5. CUPED as a variance-reduction layer before inference

Outcomes in games are often predictable from pre-experiment engagement or
spend. **Controlled-experiment Using Pre-Experiment Data (CUPED)** exploits
this by regressing the in-period outcome on pre-period covariates (with
missingness indicators) at the user level,

$$
Y_i \;=\; \beta_0 + \mathbf{x}_i^\top \boldsymbol\beta + \varepsilon_i ,
\tag{10}
$$

where:

- $\beta_0$ is the intercept (the expected outcome for a user whose
  covariates are all at the reference level / zero).
- $\boldsymbol\beta$ is the vector of covariate slopes; $\beta_j$ is the
  expected change in $Y_i$ per unit change in the $j$-th pre-experiment
  covariate, holding the others fixed.
- $\mathbf{x}_i$ is the pre-experiment covariate vector for user $i$;
  each component is observed *before* randomization and therefore cannot
  have been affected by treatment.
- $\varepsilon_i$ is the residual noise (what the covariates cannot
  explain), assumed to have zero conditional mean.

Ordinary least squares produces estimates $\hat\beta_0$ and
$\hat{\boldsymbol\beta}$, and the fitted value for user $i$ is

$$
\hat Y_i \;=\; \hat\beta_0 + \mathbf{x}_i^\top \hat{\boldsymbol\beta} .
\tag{11}
$$

Intuitively, $\hat Y_i$ is the "best guess" of user $i$'s in-period outcome
made using only pre-experiment information. CUPED then replaces the raw
outcome by the residual — the part of $Y_i$ that the pre-period covariates
could not predict —

$$
Y_i^{\text{CUPED}} \;=\; Y_i - \hat Y_i .
\tag{12}
$$

Because $\hat Y_i$ is constructed from pre-randomization variables,
subtracting it removes noise without distorting the treatment effect; in
expectation, $\mathbb{E}[Y_t^{\text{CUPED}} - Y_C^{\text{CUPED}}] = \Delta_t$.
Group means and standard errors (1)–(3) are then recomputed with
$Y_i^{\text{CUPED}}$ in place of $Y_i$, yielding smaller $s_g$ and hence
smaller $\widehat{\mathrm{SE}}(\bar Y_g)$. Only after this step does the
pipeline form $\hat\Delta_t$, $\hat\rho_t$, and $\hat\sigma_{\text{obs},t}$:
variance reduction is part of the measurement model, not an afterthought.

## 6. Bayesian inference on relative uplift

### 6.1 Normal–normal conjugate update

The parameter we want to learn in this step is the same $\rho_t$ introduced
in Section 4 — the population relative treatment effect of variant $t$ vs.
control. Its sample estimator $\hat\rho_t$, with standard error
$\hat\sigma_{\text{obs},t}$ from (9), is treated as a noisy measurement of
this unknown $\rho_t$. We do *not* introduce a separate Greek letter (such
as $\theta$) for the latent parameter; the population vs. estimator
distinction is carried entirely by the hat, so $\rho_t$ is what we want to
know and $\hat\rho_t$ is what the data tell us. The prior–likelihood pair
is:

$$
\rho_t \;\sim\; \mathcal{N}\!\bigl(\mu_0,\ \sigma_0^2\bigr) ,
\qquad
\hat\rho_t \,\big|\, \rho_t \;\sim\; \mathcal{N}\!\bigl(\rho_t,\ \hat\sigma_{\text{obs},t}^2\bigr) .
\tag{13}
$$

The normal prior and normal sampling model are conjugate. With precisions
$\tau_0 = 1/\sigma_0^2$ and $\tau_{\text{data},t} = 1/\hat\sigma_{\text{obs},t}^2$,
the posterior is $\rho_t \,|\, \hat\rho_t \sim \mathcal{N}(\mu_{\text{post},t}, \sigma_{\text{post},t}^2)$
with

$$
\tau_{\text{post},t} \;=\; \tau_0 + \tau_{\text{data},t} ,
\tag{14}
$$

$$
\mu_{\text{post},t} \;=\; \tau_{\text{post},t}^{-1}\bigl( \tau_0\, \mu_0 + \tau_{\text{data},t}\, \hat\rho_t \bigr) ,
\tag{15}
$$

$$
\sigma_{\text{post},t}^2 \;=\; \tau_{\text{post},t}^{-1} .
\tag{16}
$$

The posterior mean (15) is a precision-weighted blend of prior belief
$\mu_0$ and the data $\hat\rho_t$; the prior variance $\sigma_0^2$ controls
how much the posterior shrinks toward $\mu_0$.

### 6.2 What is not modeled jointly

Each treatment variant receives its own marginal posterior
$p(\rho_t \,|\, \text{data})$. There is no joint model over all variants
that would fully capture dependence across multiple test groups;
multi-variant statements rely on marginal posteriors and simple decision
rules below, not on a single multi-test hierarchical decision problem
across variants.

### 6.3 Decision-oriented summaries

- **Chance to beat control:**
  $P(\rho_t > 0 \,|\, \text{data}) = 1 - \Phi\!\left( -\,\mu_{\text{post},t} / \sigma_{\text{post},t} \right)$,
  the posterior probability that the true relative uplift is positive.
- **Credible intervals:** equal-tailed intervals from the posterior
  quantile function, e.g. $\mu_{\text{post},t} \pm 1.96\,\sigma_{\text{post},t}$
  for a 95% interval.

A simple winning rule selects the variant with largest posterior mean
unless all favor the control; "inverse" KPIs (lower is better) use the
analogous minimum rule.

## 7. Frequentist reference and design diagnostics

Alongside Bayesian summaries, the workflow retains frequentist objects —
group means, standard errors, and period splits — as a transparent view of
raw and adjusted series across pre-test, in-test, and post-test windows.
This dual view helps stakeholders who think in classical terms while the
primary decision language remains posterior-based for lift.

### 7.1 Sample ratio mismatch (assignment balance)

Before trusting contrasts, check whether realized assignment counts $n_g$
match the intended allocation $\{n p_g\}_{g=1}^K$ (with $\sum_g p_g = 1$;
under equal allocation $p_g = 1/K$). The Pearson chi-square goodness-of-fit
statistic is

$$
X^2 \;=\; \sum_{g=1}^{K} \frac{\bigl( n_g - n p_g \bigr)^2}{n p_g} ,
\qquad n = \sum_{g=1}^{K} n_g ,
\tag{17}
$$

where $H_0$ is that the realized split matches the intended one, and
$X^2 \overset{\text{approx.}}{\sim} \chi^2_{K-1}$ under $H_0$. A small
$p$-value flags **sample ratio mismatch**: possible issues with
randomization, eligibility filters, or tracking, rather than a treatment
effect on the KPI.

## 8. Worked example: a battle-pass offer UI test

To make the notation above concrete, we walk a small example end-to-end
using pseudo data. All numbers are invented for illustration and are
rounded for readability; the arithmetic is consistent with the formulas in
Sections 3–7.1.

**Setting.** A mobile RPG studio tests two redesigns of the battle-pass
purchase flow. Users are randomized on login into three groups ($K = 3$):

- $g = C$: existing flow (control);
- $g = 1$: new offer UI (treatment variant 1);
- $g = 2$: new offer UI with an aggressive price anchor (treatment variant 2).

The primary KPI is 14-day per-user revenue (ARPU, in USD). The
pre-experiment covariate vector $\mathbf{x}_i$ contains 14-day pre-period
ARPU, 14-day session count, account tenure in days, and corresponding
missingness indicators ($\mathbf{x}_i \in \mathbb{R}^6$).

**Assignment-balance check (Section 7.1).** The intended split is
$p_C = p_1 = p_2 = 1/3$. Realized counts are

| Group | $n_g$ |
|---|---|
| $C$ | 5{,}024 |
| $1$ | 5{,}012 |
| $2$ | 4{,}964 |
| Total $n$ | 15{,}000 |

Applying (17) with $n p_g = 5{,}000$,

$$
X^2 \;=\; \frac{(24)^2 + (12)^2 + (-36)^2}{5{,}000}
\;=\; \frac{576 + 144 + 1{,}296}{5{,}000}
\;\approx\; 0.40 .
$$

With $K - 1 = 2$ degrees of freedom, $p \approx 0.82$, so the assignment
balance check passes and we proceed.

**Raw group summaries (Section 3.2).** Suppose the raw in-period ARPU
statistics are:

| Group | $\bar Y_g$ (USD) | $s_g$ (USD) | $\widehat{\mathrm{SE}}(\bar Y_g)$ (USD) |
|---|---|---|---|
| $C$ | $5.00$ | $14.0$ | $0.197$ |
| $1$ | $5.40$ | $14.1$ | $0.199$ |
| $2$ | $5.15$ | $14.0$ | $0.199$ |

Each SE is $s_g / \sqrt{n_g}$ from (3); e.g. $14.0 / \sqrt{5{,}024} \approx 0.197$.
In particular, $\bar Y_C^{\text{raw}} = 5.00$ (this value is fixed as the
relative-uplift denominator throughout).

**CUPED step (Section 5).** We fit (10) on the pooled sample. Suppose the
OLS fit yields (rounded)

$$
\hat\beta_0 \approx 1.20, \quad
\hat\beta_{\text{pre-ARPU}} \approx 0.65, \quad
\hat\beta_{\text{sessions}} \approx 0.030, \quad
\hat\beta_{\text{tenure}} \approx 0.0025,
$$

where the three leading slopes attach to pre-period ARPU, pre-period
session count, and account tenure (the missingness-indicator coefficients
are omitted for brevity). A user with pre-period ARPU $= \$3.00$, $20$
sessions, and tenure $200$ days has fitted value

$$
\hat Y_i \;\approx\; 1.20 + 0.65 \cdot 3.00 + 0.030 \cdot 20 + 0.0025 \cdot 200 \;\approx\; 4.25 ,
$$

and residual $Y_i^{\text{CUPED}} = Y_i - \hat Y_i$. Across all users the
residuals have sample SD roughly $50\%$ smaller than the raw outcome;
assume post-CUPED $s_g^{\text{CUPED}} \approx 7.0$ in each group, giving

$$
\widehat{\mathrm{SE}}\!\bigl(\bar Y_g^{\text{CUPED}}\bigr) \;\approx\; \frac{7.0}{\sqrt{5{,}000}} \;\approx\; 0.099
\quad \text{for } g \in \{C, 1, 2\}.
$$

**Absolute and relative uplift (Section 4).** The CUPED residual
differences are (for illustration, unchanged in magnitude from the raw
differences but with much tighter SEs):

$$
\hat\Delta_1 \;=\; 0.40,\quad
\widehat{\mathrm{SE}}\!\bigl(\hat\Delta_1\bigr) \;=\; \sqrt{0.099^2 + 0.099^2} \;\approx\; 0.140 ,
$$

$$
\hat\Delta_2 \;=\; 0.15,\quad
\widehat{\mathrm{SE}}\!\bigl(\hat\Delta_2\bigr) \;\approx\; 0.140 .
$$

Using the raw control mean $\bar Y_C^{\text{raw}} = 5.00$ as the consistent
denominator from (6),

$$
\hat\rho_1 \;=\; \frac{0.40}{5.00} \;=\; 0.080 \;(8.0\%) ,
\qquad
\hat\rho_2 \;=\; \frac{0.15}{5.00} \;=\; 0.030 \;(3.0\%) .
$$

**Delta method (Section 4.3).** Plugging into (8) for variant $t = 1$, the
two partial-derivative coefficients are

$$
\frac{1}{\bar Y_C^{\text{raw}}} \;=\; \frac{1}{5.00} \;=\; 0.200 ,
\qquad
\frac{\hat\Delta_1}{\bigl(\bar Y_C^{\text{raw}}\bigr)^2} \;=\; \frac{0.40}{25.0} \;=\; 0.016 ,
$$

so

$$
\widehat{\mathrm{Var}}\!\bigl(\hat\rho_1\bigr)
\;\approx\; (0.200)^2 (0.140)^2 + (0.016)^2 (0.197)^2
\;\approx\; 7.84 \times 10^{-4} + 1.0 \times 10^{-5}
\;\approx\; 7.94 \times 10^{-4} ,
$$

$$
\hat\sigma_{\text{obs},1} \;\approx\; \sqrt{7.94 \times 10^{-4}} \;\approx\; 0.0282 \;(2.82\%) .
$$

Note the second term (denominator sensitivity) contributes less than $1.5\%$
of the variance: when the observed lift is modest, essentially all
uncertainty in $\hat\rho_t$ comes from the numerator $\hat\Delta_t$. An
analogous calculation for $t = 2$ gives $\hat\sigma_{\text{obs},2} \approx 0.0280$.

**Bayesian update (Section 6).** Adopt the (informed) prior
$\rho_t \sim \mathcal{N}(\mu_0, \sigma_0^2)$ with $\mu_0 = 0.05$ (5% prior
mean lift) and $\sigma_0 = 0.05$. Then $\tau_0 = 1/0.05^2 = 400$. For
variant $t = 1$, $\tau_{\text{data},1} = 1/0.0282^2 \approx 1{,}259$, so
from (14)–(16)

$$
\tau_{\text{post},1} \;=\; 400 + 1{,}259 \;=\; 1{,}659 ,
\qquad
\sigma_{\text{post},1} \;=\; \tau_{\text{post},1}^{-1/2} \;\approx\; 0.0246 ,
$$

$$
\mu_{\text{post},1}
\;=\; \frac{400 \cdot 0.05 + 1{,}259 \cdot 0.080}{1{,}659}
\;=\; \frac{20 + 100.7}{1{,}659}
\;\approx\; 0.0728 \;(7.28\%) .
$$

Chance to beat control: $P(\rho_1 > 0 \,|\, \text{data}) = \Phi(0.0728 / 0.0246) = \Phi(2.96) \approx 0.998$.
A 95% credible interval is $0.0728 \pm 1.96 \cdot 0.0246$, i.e.
$[0.0246,\ 0.1210]$, or roughly $[2.5\%,\ 12.1\%]$.

For variant $t = 2$ the analogous computation gives
$\mu_{\text{post},2} \approx 0.0348$, $\sigma_{\text{post},2} \approx 0.0244$,
and $P(\rho_2 > 0 \,|\, \text{data}) \approx \Phi(1.43) \approx 0.92$,
with 95% credible interval $\approx [-0.013,\ 0.083]$.

**Decision.** Both variants have positive posterior means; variant $1$ has
a materially higher one and a chance-to-beat-control above $99\%$, while
variant $2$ is at $\approx 92\%$. Under the winning rule in Section 6.3,
variant $1$ is selected for rollout; variant $2$ is flagged as
"inconclusive but directionally positive" and could be iterated on.

**What the example illustrates.** Each number above is traceable to a
single formula: $\bar Y_g$ to (1), $\widehat{\mathrm{SE}}(\bar Y_g)$ to (3),
$\hat\Delta_t$ to (4), $\hat\rho_t$ to (6), $\hat\sigma_{\text{obs},t}$ to
(8)–(9), the posterior summaries to (14)–(16), and the SRM check to (17).
The same symbols carry the same meaning wherever they appear.

## 9. Tensions with current practice and directions for improvement

The methodology above matches a practical pipeline: CUPED for measurement,
delta-method uncertainty, then conjugate Bayes on relative uplift with an
informed prior. In parallel, the following issues from ongoing practice
are worth stating explicitly for peer or course review.

**Rigid informed priors.** A common implementation choice is to encode
prior beliefs about relative lift through fixed hyperparameters (for
example, $\mu_0 = 0.05$ with $\sigma_0 = 0.05$ as in Section 8). Such a
one-size-fits-all prior can be problematic: it may reduce effective power
for true effects smaller than $\mu_0$, while strongly shrinking estimates
when the true lift is much larger, which can understate the full business
impact of a large win. A natural improvement is to replace a hard-coded
$(\mu_0, \sigma_0)$ with **dynamic priors** informed by historical
experiments, segment-level baselines, and analyst input, while documenting
sensitivity of conclusions to those choices.

**Subgroup analyses and multiple testing.** Teams often slice results by
platform, geography, or player tier to hunt for uplift in subsets. Even
when the overall experiment is well powered, exploratory subgroup
comparisons multiply the number of hypotheses and inflate the risk of
Type I error (false positives): a null overall result can still produce a
bright-looking segment by chance. Mitigations used in the wider literature
include Bonferroni-style correction across a prespecified family of tests,
hierarchical or partial pooling across segments, or preregistering a small
number of subgroup hypotheses. The core platform narrative above does
not, by itself, enforce such corrections; any improvement plan should
separate **confirmatory** (prespecified) from **exploratory** slicing and
apply error-rate control where appropriate.

## 10. Closing the loop: end-to-end logic

In summary, the methodology chains four ideas:

1. **Randomized assignment** identifies contrasts $\mu_t - \mu_C$ between
   the control group and each treatment variant at the user level.
2. **CUPED** (equations (10)–(12)) removes predictable variation so that
   group-level means carry more information per user.
3. **Delta-method propagation** (equations (7)–(9)) turns those means
   into a relative uplift $\hat\rho_t$ and a standard error
   $\hat\sigma_{\text{obs},t}$ on the business scale.
4. **Bayesian updating** (equations (13)–(16)) merges $\hat\rho_t$ with
   an informed prior on $\rho_t$, yielding posterior probabilities and
   credible intervals aligned with product language.

Assignment-balance checks (17) and frequentist tabular views sit alongside
this story: the former guards the design; the latter keeps raw evidence
visible while priors and posteriors encode judgment and uncertainty about
lift.

**Topics for external review.** Worth scrutinizing are: (i) treating
$\hat\sigma_{\text{obs},t}$ as known rather than propagating full
uncertainty from CUPED and the delta method; (ii) independence assumptions
in ratio metrics; (iii) marginal posteriors across variants and KPIs
versus joint decision rules; (iv) clustering or network effects not
captured by group-level standard errors; and (v) the prior and
multiple-testing points raised above.
