# Barebones-HAMUX
> [HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation. A temporary solution as HAMUX is being rebuilt.
> See the original [HAMUX documentation](https://bhoov.com/hamux/) for explanation.

Consists of 1 main file **`bbhamux.py`** (`<200` lines of important code) and 1 demo notebook: **`demo.ipynb`** 

**All other files are scaffolding** for e.g., docs, tests, pypi...

[The documentation](https://bhoov.com/barebones-hamux/) is designed to be a thorough but gentle introduction to everything you need to know about (energy-based) Associative Memories and Hopfield Networks.

> *`bbhamux` is pronounced as "barebones hamux" or "baby hamux".*

## Installation

**From pip**

```
pip install bbhamux
```

**From single file**

All logic is in one file: `bbhamux.py`. Copy this file into your project, modify as needed. *The best kind of research code.*

You will need to manually install dependencies:

```
pip install equinox jax
```

Install correct version of `jaxlib` for your hardware (e.g., to run on GPUs).

### Testing

```
pip install pytest
pytest test.py
```

### Quickstart

Run `demo.ipynb` for an example training on MNIST. *Works best with GPU*

## Development & Contributing

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies. 

Get started with your dev environment. After cloning:

```bash
. scripts/setup.sh # Setup the environment
. scripts/activate.sh # Activate the environment

# JAX installation:
# The base dependencies install a CPU version of JAX.
# For GPU acceleration, you need to install a specific JAX+CUDA version
# compatible with your hardware setup. Find the correct version here:
# https://jax.readthedocs.io/en/latest/installation.html#cuda-gpu-support
#
# **Important:** Run the correct `uv pip install -U "jax[cudaXX]"` command
# *after* running the setup script whenever you set up or resync your environment
# to ensure the correct JAX version is active.
#
# Example for CUDA 12:
uv pip install -U "jax[cuda12]"

# Verify your JAX installation works with your GPU:
uv run python -c "import jax; import jax.numpy as jnp; print(jax.devices('gpu')); print(jnp.ones(2) * jnp.zeros(2))"

# Check that code works
uv run pytest
```

Finally, run `demo.ipynb` using the ipython kernel `bbhamux` (installed by `setup.sh`).

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