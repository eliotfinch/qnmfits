# qnmfits
Least-squares fitting of quasinormal modes to ringdown waveforms.

## Installation

The package can be installed with `pip install .` - this assumes you are in the same directory as the `pyproject.toml` file. All dependencies for SXS waveform analysis should be installed automatically. Currently, the [`gwsurrogate`](https://pypi.org/project/gwsurrogate/) and [`surfinBH`](https://pypi.org/project/surfinBH/) packages (needed for the analysis of surrogate models) are not automatically installed.

Note that when first importing the package, there may be a short delay.

You may prefer to install the dependencies via `conda` - please see the dependencies in the `pyproject.toml` file. All packages are available on `conda` or `conda-forge`. You may additionally need to install `spinsfast` via

```bash
conda install --channel conda-forge spinsfast
```

If using `conda`, I recommend installing all dependencies via `conda` first, and then running `pip install .` in the cloned `qnmfits` directory.
