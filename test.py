from bbhamux import Neurons, HAM, VectorizedHAM, DenseSynapse, ConvSynapse

import pytest
from typing import *
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.numpy as jnp
import jax
import jax.random as jr

from bbhamux import (
    lagr_identity,
    _repu,
    lagr_repu,
    lagr_softmax,
    lagr_exp,
    _rexp,
    lagr_rexp,
    lagr_tanh,
    _tempered_sigmoid,
    lagr_sigmoid,
    _simple_layernorm,
    lagr_layernorm,
    _simple_spherical_norm,
    lagr_spherical_norm,
)

import equinox as eqx


key = jr.PRNGKey(0)
xtest = jr.normal(key, (16,)) * 5 - 3


def test_identity():
    assert jnp.allclose(xtest, jax.grad(lagr_identity)(xtest))


def test_repu():
    # Same as ReLU
    assert jnp.allclose(_repu(xtest, 1), jax.grad(lambda x: lagr_repu(x, 2))(xtest))
    # Higher degrees
    assert jnp.allclose(
        jnp.maximum(xtest, 0) ** 3, jax.grad(lambda x: lagr_repu(x, 4))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_softmax(beta: float):
    assert jnp.allclose(
        jax.nn.softmax(beta * xtest), jax.grad(lambda x: lagr_softmax(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_exp(beta: float):
    assert jnp.allclose(
        jnp.exp(beta * xtest), jax.grad(lambda x: lagr_exp(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_rexp(beta: float):
    assert jnp.allclose(
        _rexp(xtest, beta), jax.grad(lambda x: lagr_rexp(x, beta))(xtest)
    )


@pytest.mark.parametrize(
    "beta",
    [0.1, 1.0, 5.0],
)
def test_tanh(beta: float):
    assert jnp.allclose(
        jnp.tanh(beta * xtest), jax.grad(lambda x: lagr_tanh(x, beta))(xtest)
    )


@pytest.mark.parametrize("beta", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("scale", [0.1, 1.0, 5.0])
def test_tempered_sigmoid(beta: float, scale: float):
    assert jnp.allclose(
        _tempered_sigmoid(xtest, beta=beta, scale=scale),
        jax.grad(lambda x: lagr_sigmoid(x, beta=beta, scale=scale))(xtest),
    )


@pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("delta", [0.1, 1.0, 5.0])
def test_layernorm(gamma: float, delta: float):
    delta = jnp.ones_like(xtest) * delta
    assert jnp.allclose(
        _simple_layernorm(xtest, gamma=gamma, delta=delta),
        jax.grad(lambda x: lagr_layernorm(x, gamma=gamma, delta=delta))(xtest),
        rtol=1e-3,
    )


@pytest.mark.parametrize("gamma", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("delta", [0.1, 1.0, 5.0])
def test_spherical_norm(gamma: float, delta: float):
    delta = jnp.ones_like(xtest) * delta
    assert jnp.allclose(
        _simple_spherical_norm(xtest, gamma=gamma, delta=delta),
        jax.grad(lambda x: lagr_spherical_norm(x, gamma=gamma, delta=delta))(xtest),
        rtol=1e-3,
    )


neuron_shape = (5,)
beta = 3.0
neuron = Neurons(lagrangian=lambda x: lagr_softmax(x, beta=beta), shape=neuron_shape)
act_fn = lambda x: jax.nn.softmax(beta * x)


class SimpleSynapse(eqx.Module):
    W: jax.Array

    def __init__(self, key: jax.Array, shape: Tuple[int, int]):
        self.W = 0.1 * jr.normal(key, shape)

    def __call__(self, g1, g2):
        return -jnp.einsum("...d,de,...e->...", g1, self.W, g2)


@pytest.fixture
def simple_ham():
    d1, d2 = (5, 7)
    neurons = {
        "image": Neurons(lagr_identity, (d1,)),
        "hidden": Neurons(lagr_softmax, (d2,)),
    }
    synpases = {"s1": SimpleSynapse(jr.PRNGKey(0), (d1, d2))}
    connections = [
        # (vertices, hyperedge)
        (("image", "hidden"), "s1")
    ]
    ham = HAM(neurons, synpases, connections)
    return ham


def test_init():
    """Test that neuron shapes are as expected"""
    assert neuron.init().shape == neuron_shape
    assert neuron.init(bs=3).shape == (3, *neuron_shape)


def test_activations():
    """Test that activations and g are the same"""
    x = neuron.init()
    assert jnp.all(neuron.activations(x) == neuron.g(x))
    assert jnp.allclose(act_fn(x), neuron.g(x))


def test_ham_lengths(simple_ham: HAM):
    """Test configuration of our example HAM"""
    assert simple_ham.n_neurons == 2
    assert simple_ham.n_synapses == 1
    assert simple_ham.n_connections == 1


def test_vham_lengths(simple_ham: HAM):
    """Test configuration of our example HAM, vectorized"""
    vham = simple_ham.vectorize()
    assert vham.n_neurons == 2
    assert vham.n_synapses == 1
    assert vham.n_connections == 1


def test_ham_dEdg(simple_ham: HAM):
    """Test that dEdg is correct for single example"""
    xs = simple_ham.init_states()
    gs = simple_ham.activations(xs)
    auto_E, auto_dEdg = jax.value_and_grad(simple_ham.energy)(gs, xs)
    man_E, man_dEdg = simple_ham.dEdg(gs, xs, return_energy=True)

    assert jnp.allclose(auto_E, man_E)
    assert jnp.allclose(auto_dEdg["image"], man_dEdg["image"])
    assert jnp.allclose(auto_dEdg["hidden"], man_dEdg["hidden"])


def test_vectorize_unvectorize(simple_ham: HAM):
    """Test API of vectorization/unvectorization"""
    vham = simple_ham.vectorize()
    assert isinstance(vham, VectorizedHAM)
    assert isinstance(vham.vectorize(), VectorizedHAM)
    assert isinstance(vham.unvectorize(), HAM)
    assert isinstance(simple_ham.unvectorize(), HAM)


# @pytest.mark.slow
@pytest.mark.parametrize("stepsize", [0.001, 0.01, 0.1])
def test_ham_energies(simple_ham: HAM, stepsize, nsteps=10):
    """Test that single HAM decreases energy of example"""
    energies = []
    xs = simple_ham.init_states()
    xs["image"] = jr.normal(jr.PRNGKey(2), xs["image"].shape)
    for i in range(nsteps):
        gs = simple_ham.activations(xs)
        E, dEdg = simple_ham.dEdg(gs, xs, return_energy=True)
        energies.append(E)
        xs = jax.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

    Estacked = jnp.stack(energies)
    assert Estacked.shape == (nsteps,)
    assert jnp.all(jnp.diff(jnp.array(energies)) <= 0)


# @pytest.mark.slow
@pytest.mark.parametrize("stepsize", [0.001, 0.01, 0.1])
def test_vham_energies(simple_ham: HAM, stepsize, nsteps=10):
    """Test that all examples of vectorized HAM decrease in energy"""
    bs = 3
    energies = []
    xs = simple_ham.init_states(bs=bs)
    vham = simple_ham.vectorize()
    xs["image"] = jr.normal(jr.PRNGKey(1), xs["image"].shape)
    for i in range(nsteps):
        gs = vham.activations(xs)
        E, dEdg = vham.dEdg(gs, xs, return_energy=True)
        energies.append(E)
        xs = jax.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

    Estacked = jnp.stack(energies).T
    assert Estacked.shape == (bs, nsteps)
    assert jnp.all(jnp.diff(Estacked, axis=-1) <= 0)


def test_dense_synapse():
    N1, N2 = 3, 1
    g1 = jr.normal(jr.PRNGKey(0), (N1,))
    g1 = g1 / jnp.sqrt((g1 ** 2).sum())
    g2 = jnp.ones((N2,))
    syn = DenseSynapse(jr.PRNGKey(2), N1, N2)
    newW = syn.W.at[:,0].set(g1)
    syn = eqx.tree_at(lambda x: x.W, syn, newW)

    # If perfectly aligned, the energy should be -1 * magnitude of g1
    assert syn(4*g1, g2) == jnp.array(-4.)

def test_conv_synapse():
    cout, cin = 7, 3
    filter_shape = (3,3)
    im_shape = (12,12,3)
    window_strides = (3,3)
    syn = ConvSynapse(jr.PRNGKey(2), cout, cin, filter_shape, window_strides)

    g1 = jr.normal(jr.PRNGKey(0), im_shape)
    x2 = syn.forward_conv(g1[None])
    assert x2.shape == (1, 4, 4, cout)

    # Every patch contributes an energy
    # Just test that the energy is computable
    syn(g1, x2[0])