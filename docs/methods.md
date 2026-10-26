# Methods and limits

## Gaussian summary model

Let $b_{jr}$ be effect $j$ in condition $r$, and let $\widehat b_{jr}$ and $\widehat s_{jr}$ be its estimate and standard error. The vector $b_j$ contains the true effects in $R$ conditions. Following equations (2)–(4) of [Urbut et al. (2019)](https://doi.org/10.1038/s41588-018-0268-8),

$$
p(\widehat b_j\mid b_j,V_j)=N_R(\widehat b_j;b_j,V_j),
$$

$$
p(b_j\mid\pi,U)=\sum_{k=1}^{K}\sum_{l=1}^{L}\pi_{k,l}
N_R(b_j;0,\omega_l U_k).
$$

Here, $N_R$ denotes an $R$-dimensional normal distribution (a density when its covariance is positive definite), $V_j$ is the observation-error covariance, $U_k$ describes an effect-sharing pattern, $\omega_l$ multiplies its covariance, and $\pi_{k,l}$ is a mixture proportion. The proportions are nonnegative and sum to one. The covariance patterns are positive semidefinite (PSD). Consequently,

$$
p(\widehat b_j\mid\pi,U,V_j)=\sum_{k=1}^{K}\sum_{l=1}^{L}\pi_{k,l}
N_R(\widehat b_j;0,\omega_l U_k+V_j).
$$

Observation-error matrices must be positive definite. Prior patterns may be singular, including a point mass at zero. Rows are treated as independent when fitting the mixture likelihood.

## Effect scales and observation errors

`alpha=0` fits exchangeable effects (EE): the common prior describes effects in their original units, independently of the supplied standard errors. `alpha=1` fits exchangeable Z scores (EZ): the common prior instead describes effects divided by their standard errors. EZ fitting divides both the estimates and their sampling errors by those standard errors. Posterior summaries are returned in the original effect units. Original-scale log densities include the transformation Jacobian for observed coordinates.

With supplied standard errors and error correlation $C$, the model uses $V_j=S_j C S_j$, where $S_j$ has diagonal entries $\widehat s_{jr}$. `error_covariance` supplies $V_j$ directly and replaces both standard errors and correlation inputs. `estimate_null_correlation` estimates correlation from complete rows with all absolute Z scores below a threshold; selection can bias this estimate.

Missing effect estimates use `NaN`. Their standard errors remain required, including under EZ, to define the output scale. Likelihoods use observed-condition blocks of $V_j$; posterior inference can include unobserved conditions. This does not correct informative missingness. PCA and extreme-deconvolution learning require complete selected rows.

## Covariance learning and scales

Canonical patterns describe effects that, within a Gaussian component, are uncorrelated, nonzero in one condition, equal across conditions, or positively correlated. A mixture of diagonal-covariance components need not make the effects marginally independent. Principal component analysis (PCA) supplies candidate directions from strong rows. Its default uses an uncentered singular value decomposition (SVD) of working effect estimates, following the R helper. Centered Z-score PCA requires `alpha=1` and `center=True`.

Extreme deconvolution fits a zero-mean Gaussian mixture while accounting for observation error, following the noisy-mixture approach used by [Urbut et al. (2019)](https://doi.org/10.1038/s41588-018-0268-8). The algorithm can have local optima. An exactly singular start cannot expand its support under unconstrained expectation–maximization (EM) updates. `ridge` sets an explicit eigenvalue floor; `rank` limits component rank. Projected proposals are accepted only when the observed likelihood does not decrease. A rejected proposal and an iteration limit are reported separately from convergence.

Patterns are normalized by their largest diagonal entry unless `normalize=False`. Python `scales` contains standard-deviation multipliers: their squares are the paper's covariance multipliers $\omega_l$. The automatic grid spans small working standard errors through the largest observed working effect. Use an explicit grid when matching another implementation.

## Mixture fitting and uncertainty

`fit_prior` estimates mixture weights using the supplied rows. `null_weight=1` uses maximum likelihood; larger values add a log-weight penalty favoring the global-null component. A penalized fit requires that component. Inspect the convergence flag, objective trace and Karush–Kuhn–Tucker (KKT) residual in the fitted diagnostics. A small KKT residual checks the weight-optimization conditions, not the observation model.

Posterior means, credible intervals and sign probabilities condition on the selected patterns and fitted weights. They do not include all uncertainty from learning those quantities. For each effect and condition,

$$
\mathrm{lfdr}_{jr}=\Pr(b_{jr}=0\mid\widehat b_j),
\qquad
\mathrm{lfsr}_{jr}=\min\{\Pr(b_{jr}\leq0\mid\widehat b_j),
\Pr(b_{jr}\geq0\mid\widehat b_j)\}.
$$

The local false sign rate includes zero mass in either possible sign error. `lfdr` is the marginal posterior probability of an exact zero. `joint_null_probability` concerns the entire row being zero. The displayed probabilities hold fitted prior parameters fixed. These probabilities depend on the observation model and fitted prior.

## Memory and supported workflow

A full result stores component-specific posterior covariance matrices. Storage therefore grows with rows × components × conditions squared. The kernel rejects a posterior-component allocation above 256 MiB; total process memory can be higher. Use `fit_prior` for weight fitting and `prior.iter_apply(data, batch_size=...)` for ordered posterior batches. Scoring and prior fitting use likelihood-only calculations. Reduce the batch size when the number of conditions or components is large.

Retain the original condition order when reusing a prior: columns are matched positionally. Gaussian summary inference supports EE and EZ. FLASH/UDR covariance learning and GPU fitting are not implemented.

## Student-t summary errors

`fit_student_t(data, df)` replaces the Gaussian observation errors with a native
elliptical multivariate Student-t likelihood. Each row shares one latent
precision across its observed conditions. This differs from independent
univariate t errors and from converting t statistics to normal scores.

In this route, `data.error_covariance` supplies the t **scale matrix**. The
`standard_errors` input to `set_data` supplies its marginal scale factors,
not error standard deviations. The error covariance equals the scale matrix
multiplied by `df / (df - 2)` only when `df > 2`; it does not exist otherwise.
Supply finite positive degrees of freedom, either one value or one per row.

Positive numerical quadrature integrates the common precision. Refinement
checks marginal densities, first and second moments, and the marginal CDF at
zero. Inspect `quadrature_converged` and `quadrature_relative_change`; these
checks are not rigorous error bounds for every posterior quantity. Missing
conditions use observed scale-matrix blocks. Prior weights are estimated by
marginal likelihood; posterior intervals remain conditional on fitted prior
parameters. See the [SciPy t parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).

## Direct regression

`fit_regression(X, y, ...)` fits outcomes from a numeric predictor matrix. A
predictor's coefficients share a finite mixture of zero-mean Gaussian priors
across latent outcomes. Mixture weights have a Dirichlet prior and are sampled
with the coefficients. This fully Bayesian extension combines independence Metropolis–Hastings moves
with elliptical slice updates; Gaussian MASH instead optimizes marginal
mixture weights.

Predictors are not standardized automatically. Prior `scales` are absolute
coefficient SD multipliers, so predictor units matter. A fitted intercept has
Gaussian prior SD 5 and is disabled for Cox models. Outcome columns are
conditionally independent given their predictors and group effects; shared
coefficient priors do not create a residual-error covariance model.

| Family | Likelihood and fixed inputs |
| --- | --- |
| `gaussian` | Normal errors; supplied residual `sigma` |
| `binomial` | Success counts and `trials`; logistic link |
| `poisson` | Counts; log link and positive `exposure` |
| `negative_binomial` | NB2 counts; $\mathrm{Var}(Y)=\mu+\phi\mu^2$, with fixed `dispersion` $\phi>0$ |
| `multinomial` | Codes `0` through `n_categories - 1`; final category is the baseline |
| `ordinal` | Ordered codes; logistic latent variable with fixed increasing `thresholds` |
| `zero_inflated_poisson`, `zero_inflated_negative_binomial` | Logistic structural-zero process plus a count distribution |
| `hurdle_poisson`, `hurdle_negative_binomial` | Logistic zero process plus a zero-truncated positive-count distribution |
| `cox` | Breslow partial likelihood, right censoring and delayed entry |
| `weibull` | Full proportional-hazards likelihood with fixed `shape`, right censoring and delayed entry |

Here $\mu$ is the count mean. Residual SD, NB2 dispersion, ordinal thresholds
and Weibull shape are fixed inputs; defaults do not estimate them.
Ordinal default thresholds are logits of equally spaced cumulative category
fractions. Supply appropriate values when those defaults do not represent the
analysis. See [statsmodels discrete models](https://www.statsmodels.org/stable/discretemod),
[ordered models](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html)
and [proportional hazards](https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html)
for the underlying likelihood conventions.

## Groups and prediction

`groups` adds Gaussian random intercepts; `random_design` adds random slopes.
A positive-definite `random_covariance` describes their fixed covariance shape
across random-effect columns and latent outcomes. A common variance scale is
fixed or sampled with an inverse-gamma prior of shape 2 and scale 1. This is
one grouping level, not crossed or nested group structures.

Prediction averages the transformed predictor over retained draws. Supply
new-row `offset`, `trials` and `exposure` explicitly; training values are not
reused. With `groups=None`, the random effect is zero. For nonlinear links,
this is not the population-average response. Supply known group labels for
conditional group predictions, or `allow_new_groups=True` to integrate new
groups using Gaussian prior draws conditional on each retained scale.

`predict_interval` gives intervals for conditional means or survival
probabilities. `posterior_predictive` adds outcome variability. Cox response
prediction gives relative hazards; `scale="survival"` uses a draw-specific
Breslow baseline. Those intervals condition on each plug-in baseline estimate
and omit its additional sampling uncertainty. Cox future-event draws are not
implemented. Weibull `scale="response"` also returns relative hazards;
`scale="survival"` or `"cumulative_hazard"` requires prediction times.

## Sampling limits

Inspect coefficient traces, split R-hat, estimated effective sample sizes, component
switches and prior-proposal acceptance. These finite-chain diagnostics do not
prove convergence. Mixture components with singular support or little posterior
mass can be difficult to explore. Completed sampling does not establish interval
coverage or model adequacy.

Direct regression rejects missing outcomes and predictors. Survival support
covers right censoring and delayed entry only; competing risks, time-varying
covariates and interval censoring are not implemented. Count and categorical
extensions are not comparisons with R `mashr`.
