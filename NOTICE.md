# Attribution

JointShrink implements multivariate adaptive shrinkage described by Urbut, Wang,
Carbonetto and Stephens (2019), doi:10.1038/s41588-018-0268-8. Error-aware
covariance learning uses the zero-mean form of extreme deconvolution described
by Bovy, Hogg and Roweis (2011), doi:10.1214/10-AOAS439.

The native Student-t summary likelihood and outcome-regression models are
separate extensions; they are not methods from R mashr. Regression uses
Gaussian-mixture coefficient priors and elliptical slice sampling
(Murray, Adams and MacKay, 2010, https://proceedings.mlr.press/v9/murray10a.html).
Likelihood conventions follow the cited SciPy and statsmodels documentation;
these libraries' authors have not endorsed JointShrink.

R mashr 0.2.79 supplies selected Gaussian numerical reference results. Its authors have
not endorsed JointShrink. Source inspection informed parameter conventions;
the Python numerical implementation is independently written. The documentation
navigation template follows AShrink's GPL-3.0-or-later template.

The package is distributed under GPL-3.0-or-later. See LICENSE for its terms.
