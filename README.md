# JointShrink

JointShrink estimates effects jointly across tissues or conditions using
multivariate adaptive shrinkage. It learns covariance patterns from supplied
effect estimates and their standard errors, then computes posterior summaries.
A native Student-t likelihood and direct outcome-regression models extend this
workflow. Fitting uses NumPy and SciPy and does not require R.

## Installation

Install from GitHub:

```sh
pip install "git+https://github.com/nilson01/jointshrink.git"
```

From a local checkout:

```sh
pip install .
```

Matplotlib plots and pandas tables use the `plot` and `table` extras.

## Fit effects across conditions

```python
from jointshrink import set_data, fit

data = set_data(
    [[2, 1.7], [-2, -1.4], [0, .1], [.3, .5], [-.5, -.2], [1, .8]],
    standard_errors=.5,
    condition_names=["A", "B"],
)
result = fit(data, scales=[.2, 1, 2])
print(result.posterior_mean[:2].round(4))
```

```text
[[ 1.6444  1.6444]
 [-1.5111 -1.5111]]
```

This constructed example favors a shared-effect pattern. Its intervals and sign
probabilities condition on the fitted patterns and mixture weights.

![Posterior summaries for constructed inputs](https://raw.githubusercontent.com/nilson01/jointshrink/main/docs/figures/posterior-summary.png)

## Choose an analysis

| Input | Models and entry points |
| --- | --- |
| Effect estimates and standard errors | Gaussian MASH: `set_data`, `fit`, `fit_prior` |
| Effect estimates and supplied t scales | Native multivariate Student-t: `fit_student_t` |
| Outcomes and predictors | Gaussian, binomial, Poisson, negative-binomial: `fit_regression` |
| Event times and censoring indicators | Cox and Weibull proportional hazards: `fit_survival` |
| Category codes | Multinomial and ordinal: `fit_multinomial`, `fit_ordinal` |
| Counts with a separate zero process | Poisson/NB zero-inflated and hurdle wrappers |
| Grouped outcomes | Random intercepts or slopes through `groups` and `random_design` |

The Gaussian summary route follows [Urbut et al. (2019)](https://doi.org/10.1038/s41588-018-0268-8).
Direct regression samples coefficients and mixture weights under separate
Bayesian likelihood models. Check chain diagnostics before interpreting results.
Residual SD, count dispersion,
ordinal thresholds and Weibull shape are supplied and held fixed.

## Documentation and citation

[Documentation website](https://nilson01.github.io/jointshrink/)

- [Installation](https://nilson01.github.io/jointshrink/installation/), [tutorials](https://nilson01.github.io/jointshrink/tutorials/) and [grouped API](https://nilson01.github.io/jointshrink/api/).
- [Methods and limits](https://nilson01.github.io/jointshrink/methods/), [numerical reference checks](https://nilson01.github.io/jointshrink/reference-checks/).
- [Reference manual](https://nilson01.github.io/jointshrink/manual/reference/) ([PDF](https://nilson01.github.io/jointshrink/manual/jointshrink-reference.pdf)).

Cite the method paper and record the JointShrink version used.
[Numerical comparisons](https://nilson01.github.io/jointshrink/reference-checks/) describe the checked cases
and their limits.
See [NOTICE](https://github.com/nilson01/jointshrink/blob/main/NOTICE.md) and [LICENSE](https://github.com/nilson01/jointshrink/blob/main/LICENSE) for credit and license terms.
