# Barebones-HAMUX
> [HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation. A temporary solution as HAMUX is being rebuilt.
> See the original [HAMUX documentation](https://bhoov.com/hamux/) for explanation.

Consists of 1 main file `bbhamux.py` (`<200` lines of important code) and 1 demo notebook: `demo.ipynb`. Some tests are included in `test.py`.

*`bbhamux` is pronounced as "barebones hamux" or "baby hamux".*

## Install with pip

```
pip install bbhamux
```

## Install by copying `bbhamux.py`

All logic is in one file: `bbhamux.py`. Copy this file into your project, modify as needed. The best kind of research code.

You will need to manually install dependencies:

```
pip install equinox jax
pip install pytest # for tests
pip install nbdev # for docs and examples
```

Install correct version of `jaxlib` for your hardware (e.g., to run on GPUs).

## Quickstart

Run `demo.ipynb` for an example training on MNIST. *Works best with GPU*

## Contributing

We use the [`poetry`](https://python-poetry.org/docs/) package management system. Install the `nbdev` and `pytest` dependencies in addition to the main dependencies:

```
poetry install --with dev
```

All basic tests for this package are in `test.py`.

```
pytest test.py
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