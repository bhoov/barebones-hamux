# Barebones-HAMUX
> [HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation. A temporary solution as HAMUX is being rebuilt.

Build any hopfield network using energy fundamentals. See the original [HAMUX documentation](https://bhoov.com/hamux/) for explanation.

## Quick Start

This repository is a minimal implementation of HAMUX. All logic is in one file: `hamux.py`. Please copy this file into whatever project you are working on, and modify as needed.

```
pip install equinox jax
pip install pytest # for tests
```

If desired, install correct version of `jaxlib` for your hardware (e.g., to run on GPUs).

Run `demo.ipynb` for an example training on MNIST. *Works best with GPU*

## Tests

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