# Common questions

## Does analysis require R?

No. NumPy and SciPy perform analysis. R is used only for Gaussian
[reference checks](reference-checks.md).

## Should I supply effect estimates or raw outcomes?

Use `set_data` and `fit` for estimates and standard errors from an earlier
analysis. Use `fit_student_t` for a specified native t error model. Use
`fit_regression` or its family wrappers for numeric predictors and outcomes.
These routes use different likelihoods and inference; see [methods](methods.md).

## Which regression families are available?

Gaussian, binomial/logistic, Poisson, negative-binomial (NB2), multinomial,
ordinal, Poisson/NB2 zero-inflated and hurdle, Cox, and Weibull proportional
hazards. Gaussian
random intercepts or slopes can be added with one grouping factor. Residual SD,
NB2 dispersion, ordinal thresholds and Weibull shape are supplied and held fixed.

## Are credible intervals frequentist confidence intervals?

They are posterior credible intervals under the stated model. Summary-statistic
intervals condition on fitted prior parameters. Regression intervals also depend
on finite posterior sampling. Repeated-dataset coverage needs a separate study.
Cox survival intervals omit additional plug-in baseline-estimation uncertainty.

## Can values be missing?

Summary effect estimates may use `NaN`. Each row needs an observed condition;
all scale inputs must remain finite and positive. Inference uses observed
Gaussian or t blocks. Pattern learning requires complete selected rows.
Direct regression rejects missing outcomes and predictors.

## Why did a fit stop without convergence?

Covariance learning can reach an iteration limit or reject a constrained
proposal. Mixture fitting reports its objective and Karush–Kuhn–Tucker (KKT) residual. Student-t
inference also reports quadrature refinement. Regression reports chain
traces, split R-hat, effective sample sizes and component switches. Successful
execution alone does not establish convergence.

## What does prediction do for a new group?

Known group labels use their posterior effects. Unknown labels require
`allow_new_groups=True`, which draws group effects using the retained random
scale. Omitting group labels sets their effect to zero. For a nonlinear link,
this differs from averaging responses over a population of groups.

## Why is a sharing entry NaN?

No rows met the selection threshold in either condition, so its denominator
is zero. Sharing describes posterior means; it is not the posterior probability
that two effects are equal.

## How can memory use be reduced?

For Gaussian summaries, use `fit_prior` and
`prior.iter_apply(data, batch_size=...)`. Reduce batch size for many conditions
or components. Student-t inference stores quadrature components; regression
stores retained draws. Their allocation guards are not total process-memory
limits.
