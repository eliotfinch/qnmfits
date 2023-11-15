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

## Usage

The core of the package is the [`qnmfits.py`](qnmfits/qnmfits.py) file; this contains a collection of functions for performing ringdown analyses on any supplied waveform. For an overview of the available functions, see the [package tutorial notebook](examples/package_tutorial.ipynb).

Often you will want to work with SXS waveforms. 

![QNM taxonomy](examples/qnm_taxonomy.png)
