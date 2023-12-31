"""HAMUX, a minimal implementation of the Hierarchical Associative Memory

HAMUX is the skeleton of what could be an entirely new way to build DL architectures using energy blocks.

All logic is contained in this single file.
"""


import equinox as eqx
from typing import Union, Callable, Tuple, Dict, List, Optional
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


class Neurons(eqx.Module):
    """Neurons represent dynamic variables in the HAM that are evolved during inference (i.e., memory retrieval/error correction)

    They have an evolving state (created using the `.init` function) that is stored outside the neuron layer itself
    """

    lagrangian: Union[Callable, eqx.Module]
    shape: Tuple[int]

    def __init__(
        self, lagrangian: Union[Callable, eqx.Module], shape: Union[int, Tuple[int]]
    ):
        self.lagrangian = lagrangian
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape

    def activations(self, x: jax.Array) -> jax.Array:
        """Compute the activations of the neuron layer"""
        return jax.grad(self.lagrangian)(x)

    def g(self, x: jax.Array) -> jax.Array:
        """Alias for the activations"""
        return self.activations(x)

    def energy(self, g: jax.Array, x: jax.Array) -> jax.Array:
        """Assume vectorized"""
        return jnp.multiply(g, x).sum() - self.lagrangian(x)

    def init(self, bs: Optional[int] = None) -> jax.Array:
        """Return an empty state of the correct shape"""
        if bs is None or bs == 0:
            return jnp.zeros(self.shape)
        return jnp.zeros((bs, *self.shape))

    def __repr__(self: jax.Array):
        return f"Neurons(lagrangian={self.lagrangian}, shape={self.shape})"


class HAM(eqx.Module):
    """The Hierarchical Associative Memory

    A wrapper for all dynamic states (neurons) and learnable parameters (synapses) of our memory
    """

    neurons: Dict[str, Neurons]
    synapses: Dict[str, eqx.Module]
    connections: List[Tuple[Tuple, str]]

    def __init__(
        self,
        neurons: Dict[
            str, Neurons
        ],  # Neurons are the dynamical variables expressing the state of the HAM
        synapses: Dict[
            str, eqx.Module
        ],  # Synapses are the learnable relationships between dynamic variables.
        connections: List[
            Tuple[Tuple[str, ...], str]
        ],  # Connections expressed as [(['ni', 'nj'], 'sk'), ...]. Read as "Connect neurons 'ni' and 'nj' via synapse 'sk'
    ):
        """An HAM is a hypergraph that connects neurons and synapses together via connections"""
        self.neurons = neurons
        self.synapses = synapses
        self.connections = connections

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_synapses(self) -> int:
        return len(self.synapses)

    @property
    def n_connections(self) -> int:
        return len(self.connections)

    def activations(
        self,
        xs,  # The expected collection of neurons states
    ) -> Dict[str, jax.Array]:
        """Convert hidden states of each neuron into activations"""
        gs = {k: v.g(xs[k]) for k, v in self.neurons.items()}
        return gs

    def init_states(
        self,
        bs: Optional[
            int
        ] = None,  # If provided, each neuron in the HAM has this batch size
    ):
        """Initialize neuron states"""
        xs = {k: v.init(bs) for k, v in self.neurons.items()}
        return xs

    def connection_energies(
        self,
        gs: Dict[str, jax.Array],  # The collection of neuron activations
    ):
        """Get the energy for each connection"""

        def get_energy(neuron_set, s):
            mygs = [gs[k] for k in neuron_set]
            return self.synapses[s](*mygs)

        return [get_energy(neuron_set, s) for neuron_set, s in self.connections]

    def neuron_energies(self, gs, xs):
        """Return the energies of each neuron in the HAM"""
        return {k: self.neurons[k].energy(gs[k], xs[k]) for k in self.neurons.keys()}

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""
        neuron_energies = self.neuron_energies(gs, xs)
        connection_energies = self.connection_energies(gs)
        return {"neurons": neuron_energies, "connections": connection_energies}

    def energy(self, gs, xs):
        """The complete energy of the HAM"""
        energy_tree = self.energy_tree(gs, xs)
        return jtu.tree_reduce(lambda E, acc: acc + E, energy_tree, 0)

    def dEdg(self, gs, xs, return_energy=False):
        """Calculate gradient of system energy wrt activations using cute trick:

        The derivative of the neuron energy w.r.t. the activations is the neuron state itself.
        This is a property of the Legendre Transform.
        """

        def all_connection_energy(gs):
            return jtu.tree_reduce(
                lambda E, acc: acc + E, self.connection_energies(gs), 0
            )

        dEdg = jtu.tree_map(lambda x, s: x + s, xs, jax.grad(all_connection_energy)(gs))
        if return_energy:
            return self.energy(gs, xs), dEdg
        return jax.grad(self.energy)(gs, xs)

    def vectorize(self):
        """Compute new HAM with same API, except all methods expect a batch dimension"""
        return VectorizedHAM(self)

    def unvectorize(self):
        return self


class VectorizedHAM(eqx.Module):
    """Re-expose HAM API with vectorized inputs. No logic should be implemented in this class."""

    _ham: eqx.Module

    def __init__(self, ham):
        self._ham = ham

    @property
    def neurons(self):
        return self._ham.neurons

    @property
    def synapses(self):
        return self._ham.synapses

    @property
    def connections(self):
        return self._ham.connections

    @property
    def n_neurons(self):
        return self._ham.n_neurons

    @property
    def n_synapses(self):
        return self._ham.n_synapses

    @property
    def n_connections(self):
        return self._ham.n_connections

    @property
    def _batch_axes(self: HAM):
        """A helper function to tell vmap to batch along the 0'th dimension of each state in the HAM."""
        return {k: 0 for k in self._ham.neurons.keys()}

    def init_states(self, bs=None):
        """Initialize neuron states with batch size `bs`"""
        return self._ham.init_states(bs)

    def activations(self, xs):
        """Compute activations of a batch of inputs"""
        return jax.vmap(self._ham.activations, in_axes=(self._batch_axes,))(xs)

    def connection_energies(self, gs):
        """Compute energy of every connection in the HAM"""
        return jax.vmap(self._ham.connection_energies, in_axes=(self._batch_axes,))(gs)

    def neuron_energies(self, gs, xs):
        """Compute energy of every neuron in the HAM"""
        return jax.vmap(
            self._ham.neuron_energies, in_axes=(self._batch_axes, self._batch_axes)
        )(gs, xs)

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""
        return jax.vmap(
            self._ham.energy_tree, in_axes=(self._batch_axes, self._batch_axes)
        )(gs, xs)

    def energy(self, gs, xs):
        """Compute the energy of the entire HAM"""
        return jax.vmap(self._ham.energy, in_axes=(self._batch_axes, self._batch_axes))(
            gs, xs
        )

    def dEdg(self, gs, xs, return_energy=False):
        """Compute the gradient of the energy wrt the activations of the HAM"""
        return jax.vmap(
            self._ham.dEdg, in_axes=(self._batch_axes, self._batch_axes, None)
        )(gs, xs, return_energy)

    def unvectorize(self):
        """Return an HAM energy that does not work on batches of inputs"""
        return self._ham

    def vectorize(self):
        return self


# =======================
# Example Lagrangians
# =======================
"""Default lagrangian functions that correspond to commonly used non-linearities in Neural networks.

1. Lagrangians return a scalar.
2. Lagrangians are convex
3. The derivative of a lagrangian w.r.t. its input is the activation function typically used in Neural Networks.

Feel free to use these as inspiration for building your own lagrangians. They're simple enough
"""
def lagr_identity(x):
  """The Lagrangian whose activation function is simply the identity."""
  return 0.5 * jnp.power(x, 2).sum()


def _repu(x, n):
  return jnp.maximum(x, 0) ** n


def lagr_repu(x, n):  # Degree of the polynomial in the power unit
  """Rectified Power Unit of degree `n`"""
  return 1 / n * jnp.power(jnp.maximum(x, 0), n).sum()


def lagr_relu(x):
  """Rectified Linear Unit. Same as repu of degree 2"""
  return lagr_repu(x, 2)


def lagr_softmax(
  x,
  beta: float = 1.0,  # Inverse temperature
  axis: int = -1,
):  # Dimension over which to apply logsumexp
  """The lagrangian of the softmax -- the logsumexp"""
  return 1 / beta * jax.nn.logsumexp(beta * x, axis=axis, keepdims=False)


def lagr_exp(x, beta: float = 1.0):  # Inverse temperature
  """Exponential activation function, as in [Demicirgil et al.](https://arxiv.org/abs/1702.01929). Operates elementwise"""
  return 1 / beta * jnp.exp(beta * x).sum()


def _rexp(
  x,
  beta: float = 1.0,  # Inverse temperature
):
  """Rectified exponential activation function"""
  xclipped = jnp.maximum(x, 0)
  return jnp.exp(beta * xclipped) - 1


def lagr_rexp(x, beta: float = 1.0):  # Inverse temperature
  """Lagrangian of the Rectified exponential activation function"""
  xclipped = jnp.maximum(x, 0)
  return (jnp.exp(beta * xclipped) / beta - xclipped).sum()


@jax.custom_jvp
def _lagr_tanh(x, beta=1.0):
  return 1 / beta * jnp.log(jnp.cosh(beta * x))


@_lagr_tanh.defjvp
def _lagr_tanh_defjvp(primals, tangents):
  x, beta = primals
  x_dot, beta_dot = tangents
  primal_out = _lagr_tanh(x, beta)
  tangent_out = jnp.tanh(beta * x) * x_dot
  return primal_out, tangent_out


def lagr_tanh(x, beta=1.0):  # Inverse temperature
  """Lagrangian of the tanh activation function"""
  return _lagr_tanh(x, beta).sum()


@jax.custom_jvp
def _lagr_sigmoid(
  x,
  beta=1.0,  # Inverse temperature
  scale=1.0,
):  # Amount to stretch the range of the sigmoid's lagrangian
  """The lagrangian of a sigmoid that we can define custom JVPs of"""
  return scale / beta * jnp.log(jnp.exp(beta * x) + 1)


def _tempered_sigmoid(
  x,
  beta=1.0,  # Inverse temperature
  scale=1.0,
):  # Amount to stretch the range of the sigmoid
  """The basic sigmoid, but with a scaling factor"""
  return scale / (1 + jnp.exp(-beta * x))


@_lagr_sigmoid.defjvp
def _lagr_sigmoid_jvp(primals, tangents):
  x, beta, scale = primals
  x_dot, beta_dot, scale_dot = tangents
  primal_out = _lagr_sigmoid(x, beta, scale)
  tangent_out = (
    _tempered_sigmoid(x, beta=beta, scale=scale) * x_dot
  )  # Manually defined sigmoid
  return primal_out, tangent_out


def lagr_sigmoid(
  x,
  beta=1.0,  # Inverse temperature
  scale=1.0,
):  # Amount to stretch the range of the sigmoid's lagrangian
  """The lagrangian of the sigmoid activation function"""
  return _lagr_sigmoid(x, beta=beta, scale=scale).sum()


def _simple_layernorm(
  x: jnp.ndarray,
  gamma: float = 1.0,  # Scale the stdev
  delta: Union[float, jnp.ndarray] = 0.0,  # Shift the mean
  axis=-1,  # Which axis to normalize
  eps=1e-5,  # Prevent division by 0
):
  """Layer norm activation function"""
  xmean = x.mean(axis, keepdims=True)
  xmeaned = x - xmean
  denominator = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)
  return gamma * xmeaned / denominator + delta


def lagr_layernorm(
  x: jnp.ndarray,
  gamma: float = 1.0,  # Scale the stdev
  delta: Union[float, jnp.ndarray] = 0.0,  # Shift the mean
  axis=-1,  # Which axis to normalize
  eps=1e-5,  # Prevent division by 0
):
  """Lagrangian of the layer norm activation function"""
  D = x.shape[axis] if axis is not None else x.size
  xmean = x.mean(axis, keepdims=True)
  xmeaned = x - xmean
  y = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)
  return (D * gamma * y + (delta * x).sum()).sum()


def _simple_spherical_norm(
  x: jnp.ndarray,
  gamma: float = 1.0,  # Scale the stdev
  delta: Union[float, jnp.ndarray] = 0.0,  # Shift the mean
  axis=-1,  # Which axis to normalize
  eps=1e-5,  # Prevent division by 0
):
  """Spherical norm activation function"""
  xnorm = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)
  return gamma * x / xnorm + delta


def lagr_spherical_norm(
  x: jnp.ndarray,
  gamma: float = 1.0,  # Scale the stdev
  delta: Union[float, jnp.ndarray] = 0.0,  # Shift the mean
  axis=-1,  # Which axis to normalize
  eps=1e-5,  # Prevent division by 0
):
  """Lagrangian of the spherical norm activation function"""
  y = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)
  return (gamma * y + (delta * x).sum()).sum()
