import jax
import jax.numpy as jnp
from jax import vmap

# TODO: Apply a factory function to jax.jit get_neurons. 
# Requires some substantial refactoring.
from ..models import *
model = MLP(num_classes = 10, width = 100)

# JIT-compiled function that returns neuron activities for a single state on a batch of inputs
get_neurons = jax.jit(lambda state, x_batch: model.apply({'params': state.params}, x_batch, capture_intermediates=True)[1])

# JIT-compiled and vectorized (vmap) get_neurons over the leading LearnerState axis (treating x_batch as constant)
get_neurons_parallel = jax.jit(vmap(get_neurons, in_axes=(0,None)))

# JIT-compiled function that initializes a neuron_ages dict mapping each layer to a
# zero-initialized array of ints (each entry = timesteps since that neuron last fired)
# NOTE: This implementation is for an input batch size of 1, and thus, may rarely be used
@jax.jit
def initialize_neuron_ages_(neurons):
    neuron_ages = {}
    for layer in neurons['intermediates']:
        if layer == '__call__':
            continue
        layer_shape = neurons['intermediates'][layer]['__call__'][0].shape[-1]
        neuron_ages[layer] = jnp.zeros(layer_shape, dtype=int)
    return neuron_ages

# JIT-compiled and vectorized (vmap) version of initialize_neuron_ages_ over a batch of neuron dictionaries
initialize_neuron_ages_parallel = jax.jit(vmap(initialize_neuron_ages_, in_axes=(0)))

# JIT-compiled function that resets each neuron’s age to zero if it fired (value > 0), otherwise increments its age by one
# NOTE: This implementation is for an input batch size of 1, and thus, may rarely be used
@jax.jit
def increment_neuron_ages_(neurons, neuron_ages):
    # for layer in neurons['intermediates']:
    for layer in neuron_ages:
        neuron_values = neurons['intermediates'][layer]['__call__'][0]
        negative_neurons = neuron_values <= 0
        neuron_ages[layer] = (neuron_ages[layer] * negative_neurons.astype(jnp.int32)) + negative_neurons.astype(jnp.int32)
    return neuron_ages

# JIT-compiled and vectorized (vmap) version of increment_neuron_ages_ over a batch of neuron activity dicts and their age dicts
increment_neuron_ages_parallel = jax.jit(vmap(increment_neuron_ages_, in_axes=(0,0)))

# JIT-compiled function that initializes a neuron_ages dict mapping each layer to a
# zero-initialized array of ints (each entry = timesteps since that neuron last fired)
# NOTE: This implementation is for an input batch size GREATER THAN 1
@jax.jit
def initialize_neuron_ages_batch(neurons):
    neuron_ages = {}
    for layer in neurons['intermediates']:
        if layer == '__call__':
            continue
        layer_shape = neurons['intermediates'][layer]['__call__'][0].shape[1:]
        neuron_ages[layer] = jnp.zeros(layer_shape, dtype=int)
    return neuron_ages

# JIT-compiled and vectorized (vmap) version of initialize_neuron_ages_batch over a batch of neuron dictionaries
initialize_neuron_ages_batch_parallel = jax.jit(vmap(initialize_neuron_ages_batch, in_axes=(0)))

# JIT-compiled function that resets each neuron’s age to zero if it fired (value > 0), otherwise increments its age by one
# NOTE: This implementation is for an input batch size GREATER THAN 1
@jax.jit
def increment_neuron_ages_batch(neurons, neuron_ages):
    for layer in neuron_ages:
        neuron_values = neurons['intermediates'][layer]['__call__'][0]
        batch_size = neuron_values.shape[0]
        negative_neurons = jnp.all(neuron_values <= 0, axis=0)
        negative_neurons_batch_size = batch_size * negative_neurons
        neuron_ages[layer] = (neuron_ages[layer] * negative_neurons.astype(jnp.int32)) + negative_neurons_batch_size.astype(jnp.int32)
    return neuron_ages

# JIT-compiled and vectorized (vmap) version of increment_neuron_ages_batch over a batch of neuron activity dicts and their age dicts
increment_neuron_ages_batch_parallel = jax.jit(vmap(increment_neuron_ages_batch, in_axes=(0,0)))