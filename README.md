# Barebones-HAMUX
> [HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation. A temporary solution as HAMUX is being rebuilt.
> See the original [HAMUX documentation](https://bhoov.com/hamux/) for explanation.

Consists of 1 main file `bbhamux.py` (`<200` lines of important code) and 1 demo notebook: `demo.ipynb`. Some tests are included in `test.py`.

*`bbhamux` is pronounced as "barebones hamux" or "baby hamux".*

## Install: File copy (easy, for researchers)

All logic is in one file: `bbhamux.py`. Please copy this file into whatever project you are working on, and modify as needed. You will need to manually install dependencies:

```
pip install equinox jax
pip install pytest # for tests
```

Install correct version of `jaxlib` for your hardware (e.g., to run on GPUs).

## Install: Pip (easy, for developers)

```
pip install bbhamux
```

## Quickstart

Run `demo.ipynb` for an example training on MNIST. *Works best with GPU*

## Testing

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