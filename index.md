# HAMUX


<img src="https://raw.githubusercontent.com/bhoov/hamux/main/assets/header.png" alt="HAMUX logo" width="400"/>

> `bbhamux` is a minimal version of
> [`hamux`](https://github.com/bhoov/hamux) to construct Hierarchical
> Associative Memories (i.e., super-powered Hopfield Networks). It
> represents research code that is designed to be easy to use, easy to
> hack.

## Getting started

<div>

> **Note**
>
> This documentation walks through how to use the package as if
> installing from `pip`, but the main logic for this repo lives in a
> **single `.py` file** (`src/bbhamux/bbhamux.py`). See [the original
> README](https://github.com/bhoov/barebones-hamux) for instructions on
> how to just copy and use the important file.

</div>

``` bash
pip install bbhamux
```

## HAMUX is a universal abstraction for Hopfield Networks

HAMUX fully captures the the energy fundamentals of Hopfield Networks
and enables anyone to:

- 🧠 Build **DEEP** Hopfield nets

- 🧱 With modular **ENERGY** components

- 🏆 That resemble modern DL operations

**Every** architecture built using HAMUX is a *dynamical system*
guaranteed to have a *tractable energy* function that *converges* to a
fixed point. Our deep [Hierarchical Associative
Memories](https://arxiv.org/abs/2107.06446) (HAMs) have several
additional advantages over traditional [Hopfield
Networks](http://www.scholarpedia.org/article/Hopfield_network) (HNs):

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr>
<th>Hopfield Networks (HNs)</th>
<th>Hierarchical Associative Memories (HAMs)</th>
</tr>
</thead>
<tbody>
<tr>
<td>HNs are only <strong>two layers</strong> systems</td>
<td>HAMs connect <strong>any number</strong> of layers</td>
</tr>
<tr>
<td>HNs model only <strong>simple relationships</strong> between
layers</td>
<td>HAMs model <strong>any complex but differentiable operation</strong>
(e.g., convolutions, pooling, attention, <span
class="math inline">…</span>)</td>
</tr>
<tr>
<td>HNs use only <strong>pairwise synapses</strong></td>
<td>HAMs use <strong>many-body synapses</strong> (which we denote
<strong>HyperSynapses</strong>)</td>
</tr>
</tbody>
</table>

## How does HAMUX work?

> **HAMUX** is a
> [hypergraph](%60https://en.wikipedia.org/wiki/Hypergraph)\` of
> 🌀neurons connected via 🤝hypersynapses, an abstraction sufficiently
> general to model the complexity of connections used in modern AI
> architectures.

HAMUX defines two fundamental building blocks of energy: the **🌀neuron
layer** and the **🤝hypersynapse** (an abstraction of a pairwise synapse
to include many-body interactions) connected via a
[**hypergraph**](https://en.wikipedia.org/wiki/Hypergraph). It is a
fully dynamical system, where the “hidden state”
*x*<sub>*i*</sub><sup>*l*</sup> of each layer *l* (blue squares in the
figure below) is an independent variable that evolves over time. The
update rule of each layer is entirely local; only signals from a layer’s
connected synapses (red circles in the figure below) can tell the hidden
state how to change. This is shown in the following equation:

$$\tau \frac{d x\_{i}^{l}}{dt} = -\frac{\partial E}{\partial g_i^l}$$

where *g*<sub>*i*</sub><sup>*l*</sup> are the *activations* (i.e.,
non-linearities) on each neuron layer, described in the section on
[Neuron Layers](#🌀Neuron-Layers). Concretely, we implement the above
differential equation as the following discretized equation (where the
bold **x**<sub>*l*</sub> is the collection of all elements in layer
*l*’s state):

$$ \mathbf{x}\_l^{(t+1)} = \mathbf{x}\_l^{(t)} - \frac{dt}{\tau} \nabla\_{\mathbf{g}\_l}E(t)$$

HAMUX handles all the complexity of scaling this fundamental update
equation to many layers and hyper synapses. In addition, it provides a
*framework* to:

1.  Implement your favorite Deep Learning operations as a
    [HyperSynapse](https://bhoov.github.io/hamux/synapses.html)
2.  Port over your favorite activation functions as
    [Lagrangians](https://bhoov.github.io/hamux/lagrangians.html)
3.  Connect your layers and hypersynapses into a
    [HAM](https://bhoov.github.io/hamux/ham.html) (using a hypergraph as
    the data structure)
4.  Inject your data into the associative memory
5.  Automatically calculate and descend the energy given the hidden
    states at any point in time

Use these features to train any hierarchical associative memory on your
own data! All of this made possible by
[JAX](https://github.com/google/jax).

The `examples/` subdirectory contains a (growing) list of examples on
how to apply HAMUX on real data.

## Contributing to the docs

<div>

> **Warning**
>
> This README is automatically generated from `docs_src/nbs/index.qmd`
> do NOT edit it directly.

</div>

From the root of this project:

    . scripts/activate.sh # Activate the virtual environment
    cd docs_src/
    nbdev_preview # Live preview docs site
    nbdev_test # Test the code examples in the docs
    nbdev_docs # Generate static build

Merge the doc changes into `main` or `dev` (see
`/.github/workflows/pages.yml`) to deploy the docs site to the
`gh-pages` branch.
