# Barebones-HAMUX
> [HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation. A temporary solution as HAMUX is being rebuilt.
> See the original [HAMUX documentation](https://bhoov.com/hamux/) for explanation.

Consists of 1 main file **`bbhamux.py`** (`<200` lines of important code) and 1 demo notebook: **`demo.ipynb`** 

**All other files are scaffolding** for e.g., docs, tests, pypi.

[The documentation](https://bhoov.com/barebones-hamux/) is designed to be a thorough but gentle introduction to everything you need to know about (energy-based) Associative Memories and Hopfield Networks.

> *`bbhamux` is pronounced as "barebones hamux" or "baby hamux".*

## Install with pip

```
pip install bbhamux
```

## Install by copying `bbhamux.py`

All logic is in one file: `bbhamux.py`. Copy this file into your project, modify as needed. The best kind of research code.

You will need to manually install dependencies:

```
pip install equinox jax
```

Install correct version of `jaxlib` for your hardware (e.g., to run on GPUs).

## Quickstart

Run `demo.ipynb` for an example training on MNIST. *Works best with GPU*

## Testing

```
pip install pytest
pytest test.py
```

## Writing docs

```
pip install nbdev
# Edit documentation in `nbs/`
nbdev_preview
```

## Contributing

We use [`poetry`](https://python-poetry.org/docs/) to manage dependencies. Install all dependencies (including `nbdev` and `pytest`) with:

```
poetry install --with dev
```

## Citation

If this repository is useful for this work, please cite the following:

```
@inproceedings{
hoover2022universal,
title={A Universal Abstraction for Hierarchical Hopfield Networks},
author={Benjamin Hoover and Duen Horng Chau and Hendrik Strobelt and Dmitry Krotov},
booktitle={The Symbiosis of Deep Learning and Differential Equations II},
year={2022},
url={https://openreview.net/forum?id=SAv3nhzNWhw}
}
```