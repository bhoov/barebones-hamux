[HAMUX](https://github.com/bhoov/hamux) built using [equinox](https://github.com/patrick-kidger/equinox), minimal implementation

A temporary solution as HAMUX is being rebuilt.

```
conda env create -f environment.yaml
conda activate eqx-hamux
pip install -r requirements.txt
```

Update `jax[cuda11_pip]` in `requirements.txt` to match the cuda version of your system (default CUDA v11)