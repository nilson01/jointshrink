# API by analysis

All entry points are imported from `jointshrink`. Summary inputs use rows for effects and columns for conditions. Direct regression uses observations × predictors and numeric outcomes.

## Summary data

```text
set_data(effects, standard_errors=1, *, correlation=None,
         error_covariance=None, alpha=0, row_names=None, condition_names=None)
```

Returns `MashData`. `effects` has shape `(rows, conditions)` with at least two conditions; `NaN` marks a missing estimate. Standard errors broadcast to that shape and must be finite and positive. Correlation or explicit sampling covariance can be shared `(d,d)` or row-specific `(n,d,d)`. Explicit covariance cannot be combined with correlation or nondefault standard errors. `alpha` is `0` (EE) or `1` (EZ).

`MashData.subset(indices)` selects rows and preserves covariance and scale information. Input arrays are copied and exposed read-only. Row and condition names must be unique.

## Covariance patterns

| Function | Purpose and main controls |
| --- | --- |
| `canonical_covariances(data_or_dimension, *, methods=...)` | Identity, singletons, equal effects and simple heterogeneous patterns; `null` is also available |
| `validate_covariances(patterns, *, normalize=False)` | Named PSD matrices; maximum-diagonal normalization when requested |
| `select_strong(data, *, z_threshold=3)` | Row indices with an observed absolute Z score above the threshold |
| `cov_pca(data, n_components=2, *, subset=None, center=False)` | Rank-one PCs and their aggregate; complete selected rows required |
| `cov_ed(data, initial, *, subset=None, maxiter=200, tol=1e-6, ridge=0, rank=None)` | Error-aware covariance refinement; returns `CovarianceFit` |
| `make_scale_grid(data, *, factor=sqrt(2))` | Positive standard-deviation grid |
| `expand_covariances(patterns, scales, *, point_mass=True, normalize=True)` | Pattern × squared-scale combinations, deduplicated |
| `estimate_null_correlation(data, *, z_threshold=2, minimum_rows=None, regularization=0.05)` | Correlation of centered Z scores in rows below the absolute-Z threshold, shrunk toward identity |

`CovarianceFit` exposes `covariances`, `weights` and `diagnostics`. Its likelihood history and stopping reason distinguish convergence from an iteration limit or rejected proposal. `CorrelationEstimate` exposes `correlation`, selected `row_indices` and `diagnostics`.

## Gaussian summary fitting and reuse

```text
fit_prior(data, covariances=None, *, scales=None, normalize=True,
          point_mass=True, null_weight=1, initial_weights=None,
          maxiter=10000, tol=1e-8)
fit(data, covariances=None, **options)
```

Both functions fit mixture weights on all supplied rows. Omitted patterns use canonical covariances; omitted scales use the automatic grid. `scales` are SD multipliers. `fit_prior` returns `MashPrior` without allocating posterior moments. `fit` accepts the same options and returns `MashResult`. `Mash(covariances=None, **options).fit(data)` provides the configured-fit interface.

| Prior interface | Behavior |
| --- | --- |
| `MashPrior(covariances, names, weights, *, alpha=0, metadata=None)` | Construct a validated prior from already expanded covariance matrices and weights summing to one |
| `prior.apply(data)` or `apply_prior(data, prior)` | Condition on new rows without refitting |
| `prior.iter_apply(data, *, batch_size=128)` | Yield posterior batches in row order |
| `prior.score_samples(data)` or `score_samples(data, prior)` | Per-row log density in observed original units |

A reused prior requires the same condition order, number of conditions and EE/EZ scale. Stored prior covariances are already expanded; reuse does not apply a second scale grid.

## Summary posterior results

`MashResult` extends `PosteriorResult` with `prior`, `data`, `diagnostics`, per-row `log_likelihood`, and `log_bayes_factor`. The Bayes factor compares the fitted nonnull mixture, renormalized to nonnull weight one, against the global null. A result with no nonnull prior weight has log Bayes factor `-inf`.

| Property | Shape or meaning |
| --- | --- |
| `posterior_mean`, `posterior_sd` | `(n,d)` |
| `posterior_covariance` | `(n,d,d)` |
| `lfdr`, `lfsr` | `(n,d)`; marginal posterior null probabilities and local false sign rates |
| `probability_negative`, `probability_positive` | Strictly negative/positive posterior probability |
| `joint_null_probability` | `(n,)`; all conditions exactly zero |
| `row_names`, `condition_names`, `observed` | Labels and observed-coordinate mask |

| Method | Result |
| --- | --- |
| `cdf(value)` | Marginal probability of effect ≤ value |
| `quantile(probability)` | `(n,d)` for scalar probability; `(p,n,d)` for a vector |
| `interval(level=0.95)` | Equal-tailed intervals, `(n,d,2)` |
| `sample(size=1, seed=None)` | Conditional posterior draws, `(size,n,d)` |
| `contrast(coefficients, *, names=None)` | Joint posterior for linear contrasts; coefficients have shape `(contrasts,d)` |
| `significant(threshold=0.05, *, by="lfsr", any_condition=True)` | Row indices selected in any condition, or all when `False` |
| `sharing(*, factor=0.5, threshold=0.05, by="lfsr", mode="magnitude")` | Posterior-mean sharing matrix plus numerator, denominator and definition |
| `to_records(*, level=0.95)` | Long-form Python records |
| `to_dataframe(*, level=0.95)` | pandas table; requires pandas |
| `summary(*, threshold=0.05, by="lfsr")` | Compact dictionary |
| `format_summary(*, threshold=0.05, by="lfsr")` | Readable summary text |

`MashResult.shrink(data)` reuses its prior. `score_samples(data)` evaluates new rows. Uncertainty conditions on the fitted patterns and weights.

## Summary plots

| Function | Display |
| --- | --- |
| `plot_covariances(prior_or_mapping, *, axes=None, ncols=3, condition_names=None)` | Covariance patterns |
| `plot_weights(prior_or_result, *, ax=None)` | Fitted mixture weights |
| `plot_sharing(result, *, factor=0.5, threshold=0.05, by="lfsr", mode="magnitude", ax=None)` | Descriptive sharing matrix |
| `plot_intervals(result, *, condition=0, indices=None, level=0.95, ax=None)` | Marginal posterior means and intervals |
| `plot_diagnostics(result_or_covfit, *, ax=None)` | Fitting objective or covariance-learning likelihood |
| `plot_effects(result, condition=0, *, ax=None)` | Supplied estimates and posterior means |

Plots require Matplotlib and return axes. They do not change global style or save files automatically.

## Save and load

```text
save_data(data, path, *, overwrite=False)
load_data(path)
save_prior(prior, path, *, overwrite=False)
load_prior(path)
save_result(result, path, *, overwrite=False)
load_result(path)
```

These functions use schema-checked JSON rather than pickle. Existing files require `overwrite=True`. Gaussian result loading reapplies the saved prior; native $t$ result loading repeats quadrature with saved degrees of freedom. Neither refits mixture weights. Use the separate regression save/load functions below for retained regression draws.

## Student-t summary fitting

```text
fit_student_t(data, df, covariances=None, *, scales=None, normalize=True,
              point_mass=True, null_weight=1, initial_weights=None,
              quadrature_order=64, max_quadrature_order=256,
              quadrature_tol=1e-5, maxiter=10000, tol=1e-8)
apply_student_prior(data, prior, df, *, quadrature_order=64,
                    max_quadrature_order=256, quadrature_tol=1e-5)
load_student_result(path)
```

Returns `StudentTResult`, with the summary posterior methods above. `df` is
finite and positive, scalar or one value per row. `data.error_covariance` is
the t **scale matrix**, not its covariance. Its standard-error inputs specify
marginal t scales. Inspect both mixture-fit and quadrature diagnostics.

`result.shrink(data, df=...)` and `result.score_samples(data, df=...)` require
new degrees of freedom explicitly. `result.save(path)` and
`load_student_result(path)` preserve the t likelihood and repeat quadrature
without refitting weights. Generic `save_result`/`load_result` also preserve
the likelihood. `prior.apply(data)` remains a Gaussian operation; use
`apply_student_prior` for t reuse.

## Direct outcome regression

```text
fit_regression(X, y, *, family="gaussian", covariances=None,
               scales=(0.1, 0.5, 1, 2), point_mass=True,
               draws=1000, warmup=500, chains=4, seed=0,
               fit_intercept=True, offset=0, groups=None,
               random_design=None, random_covariance=None,
               random_scale=1, estimate_random_scale=True, dirichlet=1,
               coefficient_names=None, outcome_names=None,
               **likelihood_options)
```

`X` is observations × predictors, without automatic standardization. `y` is
a vector or observations × outcomes for separable families. Missing inputs are
rejected. Results retain coefficient draws with shape
`(chains, draws, predictors, latent_outcomes)`. At least two chains and four
retained draws per chain are required; those minima are interface requirements,
not recommendations for inference.

| Entry point | Family and required outcome information |
| --- | --- |
| `fit_gaussian_regression(X, y, **options)` | Continuous outcomes; supplied `sigma` defaults to 1 |
| `fit_logistic_regression(X, y, **options)` | Binary or binomial success counts; `trials` defaults to 1 |
| `fit_poisson_regression(X, y, **options)` | Nonnegative counts; positive `exposure` defaults to 1 |
| `fit_negative_binomial_regression(X, y, **options)` | NB2 counts; fixed `dispersion` defaults to 1 |
| `fit_survival(X, time, event, *, family="cox", **options)` | Cox or Weibull; event indicators are 0/1; `entry` defaults to 0 |
| `fit_multinomial(X, y, **options)` | Integer category codes; `n_categories` controls the category space |
| `fit_ordinal(X, y, **options)` | Ordered integer codes; fixed `thresholds` and `n_categories` |
| `fit_zero_inflated_poisson(X, y, **options)` | Logistic zero process and Poisson counts |
| `fit_zero_inflated_negative_binomial(X, y, **options)` | Logistic zero process and NB2 counts |
| `fit_hurdle_poisson(X, y, **options)` | Logistic zero process and positive truncated Poisson counts |
| `fit_hurdle_negative_binomial(X, y, **options)` | Logistic zero process and positive truncated NB2 counts |

All wrappers return `RegressionResult` and accept the shared fitting controls.
`make_likelihood` exposes the same 12 family strings through `FAMILIES`, with
`loglike`, `predict`, `sample` and `to_dict` methods for fixed linear predictors.
Its predictors have shape `(observations, latent_outcomes)`.

For zero-inflated and hurdle models, the first latent block contains count
predictors and the second contains zero-probability logits. Their default prior
pattern is independent across latent coordinates. Multinomial models use
`n_categories - 1` logits against the final category. Ordinal models use one
predictor and fixed increasing thresholds. Cox models disable the intercept;
Weibull shape defaults to 1 and is fixed. Other fixed nuisance parameters and
the random-effect covariance are described in [methods](methods.md#direct-regression).

`groups` contains one nonempty string label per observation. The default group
term is a random intercept. `random_design` adds columns for random slopes;
`random_covariance` has shape `(q*d, q*d)`, where `q` is the number of random-effect columns and `d` the number of latent outcomes.
It must be positive definite. `estimate_random_scale=False` holds the supplied
common SD scale fixed; otherwise its variance receives an inverse-gamma prior.

## Regression results and prediction

`RegressionResult` extends `DrawResult`. These results summarize retained
samples rather than analytic Gaussian mixtures. Their `lfdr`, `lfsr` and
`joint_null_probability` estimate posterior probabilities from the draws; finite sampling can miss small probabilities.

| Property or method | Meaning |
| --- | --- |
| `draws`, `intercept_draws`, `weight_draws` | Retained coefficient, intercept and mixture-weight samples |
| `random_draws`, `random_scale_draws` | Group-effect draws and retained common scale multipliers |
| `posterior_mean`, `posterior_sd`, `posterior_covariance` | Coefficient summaries; `coefficients` aliases the mean |
| `interval(level=0.95)`, `quantile(probability)`, `cdf(value)` | Marginal empirical posterior summaries |
| `sample(size=1, seed=None)` | Resample retained coefficient draws |
| `contrast(coefficients, *, names=None)` | Linear contrasts across latent outcomes |
| `summary()`, `format_summary()` | Numeric or readable summaries |
| `to_records(level=0.95)`, `to_dataframe(level=0.95)` | Long-form coefficient tables |
| `random_effect_summary(level=0.95)` | Group-effect records |
| `diagnostics` | Split R-hat, estimated effective sample sizes, component switches, acceptance and likelihood traces |
| `save(path, *, overwrite=False)` | Safe JSON, including retained draws |

```text
result.predict(X, *, scale="response", groups=None, random_design=None,
               offset=0, allow_new_groups=False, seed=0, **options)
result.predict_interval(X, *, level=0.95, scale="response", **options)
result.posterior_predictive(X, *, seed=0, **options)
save_regression_result(result, path, *, overwrite=False)
load_regression_result(path)
```

Prediction options include `trials`, `exposure` and survival `time` where
relevant. New-row offsets default to zero; trials and exposure default to one.
`groups=None` fixes group effects at zero. Unknown labels require
`allow_new_groups=True`, which draws their effects from the conditional group
prior. Supply `random_design` when the fitted model has random slopes.

Response predictions average transformed draws, not transformed posterior-mean
coefficients. Category predictions are category probabilities; binomial
`scale="probability"` returns success probabilities. Cox and Weibull response
predictions are relative hazards. Survival predictions require times. Cox
survival uses a draw-specific plug-in Breslow baseline and omits its additional
estimation uncertainty; Cox future-event draws are unavailable.
`predict_interval` describes conditional means or survival probabilities,
whereas `posterior_predictive` adds outcome noise. See the
[group prediction limits](methods.md#groups-and-prediction).

## Regression plots

| Function | Display |
| --- | --- |
| `plot_coefficients(result, *, condition=0, level=0.95, ax=None)` | Coefficient means and credible intervals |
| `plot_chains(result, *, term=0, condition=0, ax=None)` | Retained coefficient traces by chain |
| `plot_regression_weights(result, *, level=0.95, ax=None)` | Posterior mixture weights and intervals |
| `plot_random_effects(result, *, condition=0, level=0.95, ax=None)` | Group-effect means and intervals |

These functions require Matplotlib, return axes and do not save files
automatically. A trace plot helps assess sampling behavior; it does not certify
convergence.
