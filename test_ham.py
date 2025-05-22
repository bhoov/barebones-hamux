import pytest
from typing import *
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx

from bbhamux import Neurons, HAM, VectorizedHAM, DenseSynapse, ConvSynapse
from bbhamux import lagr_identity, lagr_softmax

key = jr.PRNGKey(0)
xtest = jr.normal(key, (16,)) * 5 - 3

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
        xs = jtu.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

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
        xs = jtu.tree_map(lambda x, dx: x - stepsize * dx, xs, dEdg)

    Estacked = jnp.stack(energies).T
    assert Estacked.shape == (bs, nsteps)
    assert jnp.all(jnp.diff(Estacked, axis=-1) <= 0)


def test_dense_synapse():
    N1, N2 = 3, 1
    g1 = jr.normal(jr.PRNGKey(0), (N1,))
    g1 = g1 / jnp.sqrt((g1**2).sum())
    g2 = jnp.ones((N2,))
    syn = DenseSynapse(jr.PRNGKey(2), N1, N2)
    newW = syn.W.at[:, 0].set(g1)
    syn = eqx.tree_at(lambda x: x.W, syn, newW)

    # If perfectly aligned, the energy should be -1 * magnitude of g1
    assert jnp.allclose(syn(4 * g1, g2), jnp.array(-4.0))


def test_conv_synapse():
    cout, cin = 7, 3
    filter_shape = (3, 3)
    im_shape = (12, 12, 3)
    window_strides = (3, 3)
    syn = ConvSynapse(jr.PRNGKey(2), cout, cin, filter_shape, window_strides)

    g1 = jr.normal(jr.PRNGKey(0), im_shape)
    x2 = syn.forward_conv(g1[None])
    assert x2.shape == (1, 4, 4, cout)

    # Every patch contributes an energy
    # Just test that the energy is computable
    syn(g1, x2[0])
