# Installation

Python 3.10 or later is required.

```shell
pip install jointshrink
```

For Matplotlib plots and pandas tables:

```shell
pip install "jointshrink[plot,table]"
```

## From GitHub

```shell
python -m pip install "git+https://github.com/nilson01/jointshrink.git@main"
```

## From a local checkout

Run this command in the directory containing `pyproject.toml`:

```shell
python -m pip install .
```

NumPy and SciPy are required dependencies. Matplotlib is required only for plots; pandas is required for `to_dataframe()` tables. No R installation is needed for analysis.

## Import and fit

```python
from jointshrink import set_data, fit

data = set_data([[1.2, 1.0], [-0.8, -1.0], [0.1, -0.2]], 0.5)
result = fit(data, scales=[0.25, 0.5, 1.0, 2.0])
print(result.format_summary())
```
