# Development

## Checks

From a source checkout:

```shell
python -m pip install ".[dev,plot,table]"
python -m pytest
python -m build
```

Tests cover Gaussian conditioning, mixture fitting, covariance learning,
missing-condition inference, Student-t quadrature, all 12 regression
likelihoods, conditional prediction, persistence and plots. Independent
regression posterior checks use conjugate Gaussian cases and allow estimated
Monte Carlo error. They do not establish general calibration or convergence.

Platform evidence must identify the interpreter, operating system and source
revision. The [installed-R checks](reference-checks.md) cover selected Gaussian
summary cases and are separate from the normal Python test suite. R is not a
runtime dependency.

## Dependencies and license

NumPy and SciPy are runtime dependencies. Matplotlib and pandas support plotting and tables. Documentation uses MkDocs and MathJax. See [LICENSE](https://github.com/nilson01/jointshrink/blob/main/LICENSE) for GPL-3.0-or-later terms.
