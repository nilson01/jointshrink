# Contributing

Describe a bug with a small example, expected behavior, package version and
operating system. Remove private data before sharing an example.

For a code change, add a focused regression test and update the affected
reference or tutorial. Install the development extra and run:

```sh
pip install '.[dev,plot,table]'
python -m pytest
python -m ruff check src tests tools examples
```

Numerical changes need an independent calculation or an aligned reference case.
Passing a test suite alone does not establish statistical calibration.
