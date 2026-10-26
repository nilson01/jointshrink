# Numerical reference checks

The optional checker compares the Python workflow with installed R [`mashr`](https://github.com/stephenslab/mashr) 0.2.79. Fixed cases cover EE and EZ, supplied and estimated mixture weights, null-biased fitting, and explicit scale expansion. Compared outputs include posterior means, standard deviations, covariances, local false sign rates and local false discovery rates. An uncentered PCA comparison checks the matched helper convention.

The reference export records package versions and library origin. The checker uses absolute and relative tolerances of `2e-10` for fixed-prior cases and `2e-5` for fitted-weight cases. PCA matrices use `1e-12`. See `tools/export_mashr_reference.R` and `tools/check_mashr_reference.py` in the repository.

## Reproduce the comparison

With `mashr` and `jsonlite` available in an isolated R library, run from the repository root:

```shell
Rscript tools/export_mashr_reference.R reference.json
python tools/check_mashr_reference.py reference.json
```

R is not a runtime dependency of the Python package. The checker reads a fresh export instead of treating an old success report as current evidence.

## Differences and limits

| Area | Interpretation |
| --- | --- |
| Missing conditions | Python uses exact observed-block Gaussian likelihoods; R's large-SE substitution is an approximation |
| PCA | Match EE/EZ preprocessing, centering and selected rows before comparing matrices |
| Scales | Python inputs are SD multipliers; covariance multipliers are their squares |
| Covariance learning | Independent analytic checks cover noisy-mixture updates; an installed-R extreme-deconvolution comparison is not claimed |
| Uncertainty | Numerical agreement does not establish repeated-dataset coverage or include all hyperparameter uncertainty |

R `mashr` converts finite-degrees-of-freedom inputs to normal scores. JointShrink instead provides a separate native multivariate Student-t likelihood, checked against independent convolution integrals and distribution formulas. The installed-R comparisons above apply only to Gaussian summaries. See [methods and limits](methods.md).
