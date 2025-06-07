import jax
import jax.numpy as jnp
from jax import vmap

from ..models import *

model = MLP(num_classes = 10, width = 100)

num_classes = 10
model_CNN = CNN(num_classes = num_classes)

def initialize_neuron_ages_hist(neuron_ages, tau_max):
    neuron_ages_hist = {}
    for layer in neuron_ages:
        # num_neurons = neuron_ages[layer].shape[0]
        num_neurons = neuron_ages[layer].shape[-1]
        # Length of the histogram is tau_max + 1
        neuron_ages_hist[layer] = jnp.zeros((num_neurons, tau_max+1), dtype=int)
    return neuron_ages_hist

def initialize_neuron_ages_hist_parallel(neuron_ages, tau_max):
    return vmap(initialize_neuron_ages_hist, in_axes=(0,None))(neuron_ages, tau_max)

@jax.jit
def reinitialize_tau_indiv(state, neuron_ages):
    threshold = state.threshold
    new_threshold = {}
    for layer in neuron_ages:
        new_threshold[layer] = threshold * jnp.ones(neuron_ages[layer].shape[-1], dtype = int)
    return LearnerState.replace(state, threshold=new_threshold)

@jax.jit
def reinitialize_tau_layer(state, neuron_ages):
    threshold = state.threshold
    new_threshold = {}
    for layer in neuron_ages:
        new_threshold[layer] = threshold
    return LearnerState.replace(state, threshold=new_threshold)  

reinitialize_tau_indiv_parallel = jax.jit(vmap(reinitialize_tau_indiv, in_axes=(0,0))) 

reinitialize_tau_layer_parallel = jax.jit(vmap(reinitialize_tau_layer, in_axes=(0,0)))


# Update tau, all information should be in state
@jax.jit
def update_tau_(state, neuron_ages_hist):
    # Load LearnerState variables from state
    params = state.params
    all_thresholds = state.threshold
    reg_str = state.reg_str
    reg_params = state.reg_params
    algorithm = state.algorithm
    threshold_reset_freq = state.threshold_reset_freq
    threshold_percentile = state.threshold_percentile
    threshold_expansion_factor = state.threshold_expansion_factor
    batch_size = 100

    # New, or, updated threshold 
    new_threshold = {}

    for layer in all_thresholds:

        # Get the current layer of neuron_ages_hist and all_thresholds
        hist = neuron_ages_hist[layer]
        threshold = all_thresholds[layer]
        
        # Get tau_max from the shape of the histogram, bounds how much we can expand tau
        # TODO: potentially make tau_max a variable in LearnerState
        # Length of the histogram is tau_max + 1
        tau_max = hist.shape[1] - 1

        # Turn each neuron's histogram into a CDF and compute its threshold_percentile index
        cdf = jnp.cumsum(hist[:,1:], axis=1)
        cdf_normalized = cdf / jnp.sum(hist[:,1:], axis=1, keepdims=True)
        threshold_percentile_index = jnp.argmax(cdf_normalized >= threshold_percentile, axis=1) + 1

        # If the threshold_percentile index is below tau, contract tau to the threshold_percentile index
        # other expand tau by threshold_expansion_factor
        new_threshold_layer = jnp.where(threshold_percentile_index < threshold, threshold_percentile_index, threshold_expansion_factor*threshold)

        # Ensure that we do not exceed the tau_max nor hit 0
        new_threshold[layer] = jnp.maximum(batch_size, jnp.minimum(new_threshold_layer, tau_max))

        # reset neuron_ages_hist
        neuron_ages_hist[layer] = jnp.zeros(hist.shape, dtype = int)

    # Update the LearnerState, state
    new_state = LearnerState.replace(state, threshold = new_threshold)
    return new_state, neuron_ages_hist

@jax.jit
def update_tau(state, neuron_ages_hist, task):
    condition = task % state.threshold_reset_freq == 0
    return lax.cond(condition, update_tau_, lambda x,y: (x,y), state, neuron_ages_hist)

update_tau_parallel = jax.jit(jax.vmap(update_tau, in_axes=(0,0,None)))


###############################################################################################
# Increment Neuron Ages with the neuron ages histogram
###############################################################################################

@jax.jit
def increment_neuron_ages_snr(neurons, neuron_ages, neuron_ages_hist):
    # for layer in neurons['intermediates']:
    for layer in neuron_ages:
        neuron_values = neurons['intermediates'][layer]['__call__'][0]
        batch_size = neuron_values.shape[0]
        negative_neurons = jnp.all(neuron_values <= 0, axis=0)
        negative_neurons_batch_size = batch_size * negative_neurons

        # Update neuron_ages_hist
        # compute mask of where neurons are firing
        row_mask = jnp.logical_not(negative_neurons)
        # col_mask: a map from each neuron to its age before firing
        col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
        row_mask = row_mask[:,None]

        hist_len = neuron_ages_hist[layer].shape[1]
        indices = jnp.arange(hist_len)[None, :]

        temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
        neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

        # Finally update ages
        neuron_ages[layer] = (neuron_ages[layer] * negative_neurons.astype(jnp.int32)) + negative_neurons_batch_size.astype(jnp.int32)

    return neuron_ages, neuron_ages_hist

increment_neuron_ages_snr_parallel = jax.jit(jax.vmap(increment_neuron_ages_snr, in_axes=(0,0,0)))

###############################################################################################
# New Content Implementing Support functions for snr with a CNN
###############################################################################################

# Neuron ages are now of shape (# neurons), which is what the MLP does 
# Previously, for a CNN this was (# x-pixels, # y-pixels, # neurons)
# This requires the creation of increment_neuron_ages_snr_new 
@jax.jit
def initialize_neuron_ages_snr(neurons):
    neuron_ages = {}
    for layer in neurons['intermediates']:
        if layer == '__call__':
            continue
        layer_shape = neurons['intermediates'][layer]['__call__'][0].shape[-1]
        neuron_ages[layer] = jnp.zeros(layer_shape, dtype=int)
    return neuron_ages

initialize_neuron_ages_snr_parallel = jax.jit(vmap(initialize_neuron_ages_snr, in_axes=(0)))

# I believe this should work with my MLP, since it works for dense layers in the CNN
# but it assumes that neuron ages are of shape (# neurons)
# Also assumes that neurons are neuron-activities over a batched input: that is,
# MLP: (batch_size, # neurons), CNN: (batch_size, x-pixels, y-pixles, # neurons)
@jax.jit
def increment_neuron_ages_snr_new(neurons, neuron_ages, neuron_ages_hist):
    # for layer in neurons['intermediates']:
    for layer in neuron_ages:
        neuron_values = neurons['intermediates'][layer]['__call__'][0]
        batch_size = neuron_values.shape[0]

        all_axes_but_last = tuple(range(neuron_values.ndim - 1))
        negative_neurons = jnp.all(neuron_values <= 0, axis=all_axes_but_last)
        negative_neurons_batch_size = batch_size * negative_neurons

        # Update neuron_ages_hist
        # compute mask of where neurons are firing
        row_mask = jnp.logical_not(negative_neurons)
        # col_mask: a map from each neuron to its age before firing
        col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
        row_mask = row_mask[:,None]

        hist_len = neuron_ages_hist[layer].shape[1]
        indices = jnp.arange(hist_len)[None, :]

        temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
        neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

        # Finally update ages
        neuron_ages[layer] = (neuron_ages[layer] * negative_neurons.astype(jnp.int32)) + negative_neurons_batch_size.astype(jnp.int32)

    return neuron_ages, neuron_ages_hist

increment_neuron_ages_snr_new_parallel = jax.jit(jax.vmap(increment_neuron_ages_snr_new, in_axes=(0,0,0)))

###############################################################################################
# New Content for Refactoring so that we can merge with the main branch
###############################################################################################

@jax.jit
def reset_neurons_snr_(state, neuron_ages, neuron_ages_hist, key, x):
    # Initialize random params
    params_rand = model.init(key, x)["params"]
    # Load params from state
    params = state.params
    all_thresholds = state.threshold
    reg_str = state.reg_str
    reg_params = state.reg_params
    algorithm = state.algorithm
    threshold_reset_freq = state.threshold_reset_freq
    threshold_percentile = state.threshold_percentile
    threshold_expansion_factor = state.threshold_expansion_factor

    # Get layers from neuron_ages
    layers = []
    for layer in neuron_ages:
        layers.append(layer)

    for i in range(len(layers)-1):
        layer = layers[i]
        next_layer = layers[i+1]

        # Get threshold for current layer
        threshold = all_thresholds[layer]

        # Identify neurons to reset: conditioning on Conv_ or Dense_ layer
        reset_mask = (neuron_ages[layer] >= threshold)

        # TODO: Move this to a separate function
        # Update neuron_ages_hist
        # compute mask of where neurons are firing
        row_mask = reset_mask
        # col_mask: a map from each neuron to its age before firing
        col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
        row_mask = row_mask[:,None]

        hist_len = neuron_ages_hist[layer].shape[1]
        indices = jnp.arange(hist_len)[None, :]

        temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
        neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

        # Reset ages, but only after updating the histogram
        neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
        reset_mask = reset_mask.flatten()

        # MLP to MLP SGD Standard Reset based off reset_mask
        # Reset bias terms to zero
        params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

        # Reset incoming neuron weights according to initial distribution
        params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))
        
        # Reset outgoing weights to zero
        params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']
        
    return state.replace(params=params), neuron_ages, neuron_ages_hist

@jax.jit
def reset_neurons_snr(state, neuron_ages, neuron_ages_hist, key, x):
    all_thresholds = state.threshold
    layers = []
    reset = 0
    for layer in neuron_ages:
        layers.append(layer)
    for layer in layers[0:-1]:
        threshold = all_thresholds[layer]
        cond = jnp.max(neuron_ages[layer] >= threshold)
        reset = lax.cond(cond, (lambda x: 1), (lambda x: x), reset)
    return lax.cond(reset > 0, reset_neurons_snr_, (lambda v,w,x,y,z: (v,w,x)), state, neuron_ages, neuron_ages_hist, key, x)

reset_neurons_snr_parallel = jax.jit(vmap(reset_neurons_snr, in_axes=(0,0,0,None,None)))

##########################################################################################
###### ADAM UPDATE                                                                  ######
##########################################################################################

@jax.jit
def reset_neurons_snr_adam_(state, neuron_ages, neuron_ages_hist, key, x):
    # Initialize random params
    params_rand = model.init(key, x)["params"]
    # Load params from state
    params = state.params
    all_thresholds = state.threshold
    reg_str = state.reg_str
    reg_params = state.reg_params
    algorithm = state.algorithm
    opt_state = state.opt_state
    threshold_reset_freq = state.threshold_reset_freq
    threshold_percentile = state.threshold_percentile
    threshold_expansion_factor = state.threshold_expansion_factor

    # Get layers from neuron_ages
    layers = []
    for layer in neuron_ages:
        layers.append(layer)

    for i in range(len(layers)-1):
        layer = layers[i]
        next_layer = layers[i+1]

        # Get threshold for current layer
        threshold = all_thresholds[layer]

        # Identify neurons to reset: conditioning on Conv_ or Dense_ layer
        reset_mask = (neuron_ages[layer] >= threshold)

        # TODO: Move this to a separate function
        # Update neuron_ages_hist
        # compute mask of where neurons are firing
        row_mask = reset_mask
        # col_mask: a map from each neuron to its age before firing
        col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
        row_mask = row_mask[:,None]

        hist_len = neuron_ages_hist[layer].shape[1]
        indices = jnp.arange(hist_len)[None, :]

        temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
        neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

        # Reset ages, but only after updating the histogram
        neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
        reset_mask = reset_mask.flatten()

        # MLP to MLP Adam Standard Reset based off reset_mask
        # Reset bias terms to zero
        params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
        opt_state[0].mu[layer]['bias'] = opt_state[0].mu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
        opt_state[0].nu[layer]['bias'] = opt_state[0].nu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

        # Reset incoming neuron weights according to initial distribution
        params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))
        opt_state[0].mu[layer]['kernel'] = (opt_state[0].mu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))
        opt_state[0].nu[layer]['kernel'] = (opt_state[0].nu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))
        
        # Reset outgoing weights to zero
        params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']
        opt_state[0].mu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].mu[next_layer]['kernel']
        opt_state[0].nu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].nu[next_layer]['kernel']

    return state.replace(params=params, opt_state = opt_state), neuron_ages, neuron_ages_hist

@jax.jit
def reset_neurons_snr_adam(state, neuron_ages, neuron_ages_hist, key, x):
    all_thresholds = state.threshold
    layers = []
    reset = 0
    for layer in neuron_ages:
        layers.append(layer)
    for layer in layers[0:-1]:
        threshold = all_thresholds[layer]
        cond = jnp.max(neuron_ages[layer] >= threshold)
        reset = lax.cond(cond, (lambda x: 1), (lambda x: x), reset)
    return lax.cond(reset > 0, reset_neurons_snr_adam_, (lambda v,w,x,y,z: (v,w,x)), state, neuron_ages, neuron_ages_hist, key, x)

reset_neurons_snr_adam_parallel = jax.jit(vmap(reset_neurons_snr_adam, in_axes=(0,0,0,None,None)))


###############################################################################################
# CNN with SGD
###############################################################################################

@jax.jit
def reset_neurons_snr_CNN_(state, neuron_ages, neuron_ages_hist, key, x):
    # Initialize random params
    params_rand = model_CNN.init(key, x)["params"]
    # Load params from state
    params = state.params
    all_thresholds = state.threshold
    reg_str = state.reg_str
    reg_params = state.reg_params
    algorithm = state.algorithm
    threshold_reset_freq = state.threshold_reset_freq
    threshold_percentile = state.threshold_percentile
    threshold_expansion_factor = state.threshold_expansion_factor

    ########################
    # Conv_0 to Conv_1
    ########################
    layer = 'Conv_0'
    next_layer = 'Conv_1'
    
    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)
    reset_mask_extended = reset_mask.reshape((1,) * (neuron_ages[layer].ndim - reset_mask.ndim) + reset_mask.shape)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages, Conv_
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask_extended.astype(jnp.int32))

    # Reset incoming bias to zero, Conv_
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming kernel to prior, Conv_
    params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask[None,None,None,:].astype(jnp.int32))
    # params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask_extended.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask_extended.astype(jnp.int32))

    # Reset outgoing kernel to zero, Conv_ to Conv_
    params[next_layer]['kernel'] = (1 - reset_mask[None,None,:,None].astype(jnp.int32)) * params[next_layer]['kernel']

    ########################
    # Conv_1 to Dense_0
    ########################
    layer = 'Conv_1'
    next_layer = 'Dense_0'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)
    reset_mask_extended = reset_mask.reshape((1,) * (neuron_ages[layer].ndim - reset_mask.ndim) + reset_mask.shape)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages, Conv_
    # neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask_extended.astype(jnp.int32))

    # Reset incoming bias to zero, Conv_
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming kernel to prior, Conv_
    params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask[None,None,None,:].astype(jnp.int32))

    # Reset outgoing kernel to zero, Conv_ to Dense_
    reset_mask_ = jnp.ones((1,5,5,16)) * reset_mask.astype(jnp.int32)
    reset_mask_ = reset_mask_.reshape(-1, reset_mask_.shape[0])
    params[next_layer]['kernel'] = (1 - reset_mask_.astype(jnp.int32)) * params[next_layer]['kernel']

    ########################
    # Dense_0 to Dense_1
    ########################
    layer = 'Dense_0'
    next_layer = 'Dense_1'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
    reset_mask = reset_mask.flatten()

    # Reset bias terms to zero
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming neuron weights according to initial distribution
    params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))

    # Reset outgoing weights to zero
    params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']

    ########################
    # Dense_1 to Dense_2
    ########################
    layer = 'Dense_1'
    next_layer = 'Dense_2'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
    reset_mask = reset_mask.flatten()

    # Reset bias terms to zero
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming neuron weights according to initial distribution
    params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))

    # Reset outgoing weights to zero
    params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']

    return state.replace(params=params), neuron_ages, neuron_ages_hist
    

@jax.jit
def reset_neurons_snr_CNN(state, neuron_ages, neuron_ages_hist, key, x):
    all_thresholds = state.threshold
    layers = []
    reset = 0
    for layer in neuron_ages:
        layers.append(layer)
    for layer in layers[0:-1]:
        threshold = all_thresholds[layer]
        cond = jnp.max(neuron_ages[layer] >= threshold)
        reset = lax.cond(cond, (lambda x: 1), (lambda x: x), reset)
    return lax.cond(reset > 0, reset_neurons_snr_CNN_, (lambda v,w,x,y,z: (v,w,x)), state, neuron_ages, neuron_ages_hist, key, x)

reset_neurons_snr_CNN_parallel = jax.jit(vmap(reset_neurons_snr_CNN, in_axes=(0,0,0,None,None)))

##########################################################################################
###### CNN ADAM UPDATE                                                              ######
##########################################################################################

@jax.jit
def reset_neurons_snr_CNN_Adam_(state, neuron_ages, neuron_ages_hist, key, x):
    # Initialize random params
    params_rand = model_CNN.init(key, x)["params"]
    # Load params from state
    params = state.params
    all_thresholds = state.threshold
    reg_str = state.reg_str
    reg_params = state.reg_params
    algorithm = state.algorithm
    threshold_reset_freq = state.threshold_reset_freq
    threshold_percentile = state.threshold_percentile
    threshold_expansion_factor = state.threshold_expansion_factor
    opt_state = state.opt_state

    ########################
    # Conv_0 to Conv_1
    ########################
    layer = 'Conv_0'
    next_layer = 'Conv_1'
    
    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)
    reset_mask_extended = reset_mask.reshape((1,) * (neuron_ages[layer].ndim - reset_mask.ndim) + reset_mask.shape)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages, Conv_
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask_extended.astype(jnp.int32))

    # Reset incoming bias to zero, Conv_
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['bias'] = opt_state[0].mu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].nu[layer]['bias'] = opt_state[0].nu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming kernel to prior, Conv_
    params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask[None,None,None,:].astype(jnp.int32))
    # params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask_extended.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask_extended.astype(jnp.int32))
    opt_state[0].mu[layer]['kernel'] = (opt_state[0].mu[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32)))
    opt_state[0].nu[layer]['kernel'] = (opt_state[0].nu[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32)))

    # Reset outgoing kernel to zero, Conv_ to Conv_
    params[next_layer]['kernel'] = (1 - reset_mask[None,None,:,None].astype(jnp.int32)) * params[next_layer]['kernel']
    opt_state[0].mu[next_layer]['kernel'] = (1 - reset_mask[None,None,:,None].astype(jnp.int32)) * opt_state[0].mu[next_layer]['kernel']
    opt_state[0].nu[next_layer]['kernel'] = (1 - reset_mask[None,None,:,None].astype(jnp.int32)) * opt_state[0].nu[next_layer]['kernel']

    ########################
    # Conv_1 to Dense_0
    ########################
    layer = 'Conv_1'
    next_layer = 'Dense_0'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)
    reset_mask_extended = reset_mask.reshape((1,) * (neuron_ages[layer].ndim - reset_mask.ndim) + reset_mask.shape)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages, Conv_
    # neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask_extended.astype(jnp.int32))

    # Reset incoming bias to zero, Conv_
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['bias'] = opt_state[0].mu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].nu[layer]['bias'] = opt_state[0].nu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming kernel to prior, Conv_
    params[layer]['kernel']  = (params[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask[None,None,None,:].astype(jnp.int32))
    opt_state[0].mu[layer]['kernel'] = (opt_state[0].mu[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32)))
    opt_state[0].nu[layer]['kernel'] = (opt_state[0].nu[layer]['kernel'] * (1 - reset_mask[None,None,None,:].astype(jnp.int32)))

    # Reset outgoing kernel to zero, Conv_ to Dense_
    reset_mask_ = jnp.ones((1,5,5,16)) * reset_mask.astype(jnp.int32)
    reset_mask_ = reset_mask_.reshape(-1, reset_mask_.shape[0])
    params[next_layer]['kernel'] = (1 - reset_mask_.astype(jnp.int32)) * params[next_layer]['kernel']
    opt_state[0].mu[next_layer]['kernel'] = (1 - reset_mask_.astype(jnp.int32)) * opt_state[0].mu[next_layer]['kernel']
    opt_state[0].nu[next_layer]['kernel'] = (1 - reset_mask_.astype(jnp.int32)) * opt_state[0].nu[next_layer]['kernel']

    ########################
    # Dense_0 to Dense_1
    ########################
    layer = 'Dense_0'
    next_layer = 'Dense_1'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
    reset_mask = reset_mask.flatten()

    # Reset bias terms to zero
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['bias'] = opt_state[0].mu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].nu[layer]['bias'] = opt_state[0].nu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming neuron weights according to initial distribution
    params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['kernel'] = (opt_state[0].mu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))
    opt_state[0].nu[layer]['kernel'] = (opt_state[0].nu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))

    # Reset outgoing weights to zero
    params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']
    opt_state[0].mu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].mu[next_layer]['kernel']
    opt_state[0].nu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].nu[next_layer]['kernel']

    ########################
    # Dense_1 to Dense_2
    ########################
    layer = 'Dense_1'
    next_layer = 'Dense_2'

    # Get threshold for current layer
    threshold = all_thresholds[layer]

    # Identify neurons to reset: Conv_ layer
    reset_mask = (neuron_ages[layer] >= threshold)

    # TODO: Move this to a separate function
    # Update neuron_ages_hist
    # compute mask of where neurons are firing
    row_mask = reset_mask
    # col_mask: a map from each neuron to its age before firing
    col_mask = jnp.where(row_mask, neuron_ages[layer], 0)[:,None]
    row_mask = row_mask[:,None]

    hist_len = neuron_ages_hist[layer].shape[1]
    indices = jnp.arange(hist_len)[None, :]

    temp_neuron_ages_hist = jnp.where(indices == col_mask, row_mask, 0)
    neuron_ages_hist[layer] = neuron_ages_hist[layer] + temp_neuron_ages_hist 

    # Reset ages, but only after updating the histogram
    # Reset ages
    neuron_ages[layer] = neuron_ages[layer] * (1 - reset_mask.astype(jnp.int32))
    reset_mask = reset_mask.flatten()

    # Reset bias terms to zero
    params[layer]['bias'] = params[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['bias'] = opt_state[0].mu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))
    opt_state[0].nu[layer]['bias'] = opt_state[0].nu[layer]['bias'] * (1 - reset_mask.astype(jnp.int32))

    # Reset incoming neuron weights according to initial distribution
    params[layer]['kernel'] = (params[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32))) + (params_rand[layer]['kernel'] * reset_mask.astype(jnp.int32))
    opt_state[0].mu[layer]['kernel'] = (opt_state[0].mu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))
    opt_state[0].nu[layer]['kernel'] = (opt_state[0].nu[layer]['kernel'] * (1 - reset_mask.astype(jnp.int32)))

    # Reset outgoing weights to zero
    params[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * params[next_layer]['kernel']
    opt_state[0].mu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].mu[next_layer]['kernel']
    opt_state[0].nu[next_layer]['kernel'] = (1 - reset_mask.astype(jnp.int32))[:,None] * opt_state[0].nu[next_layer]['kernel']

    return state.replace(params=params, opt_state = opt_state), neuron_ages, neuron_ages_hist
    

@jax.jit
def reset_neurons_snr_CNN_Adam(state, neuron_ages, neuron_ages_hist, key, x):
    all_thresholds = state.threshold
    layers = []
    reset = 0
    for layer in neuron_ages:
        layers.append(layer)
    for layer in layers[0:-1]:
        threshold = all_thresholds[layer]
        cond = jnp.max(neuron_ages[layer] >= threshold)
        reset = lax.cond(cond, (lambda x: 1), (lambda x: x), reset)
    return lax.cond(reset > 0, reset_neurons_snr_CNN_Adam_, (lambda v,w,x,y,z: (v,w,x)), state, neuron_ages, neuron_ages_hist, key, x)

reset_neurons_snr_CNN_Adam_parallel = jax.jit(vmap(reset_neurons_snr_CNN_Adam, in_axes=(0,0,0,None,None)))

# Return the appropriate reset_snr_MODEL_OPTIMIZER_parallel given the model and optimizer
def get_reset_snr_parallel(model, optimizer):
    if (type(model) == MLP) & (optimizer == optax.sgd):
        return reset_neurons_snr_parallel
    if (type(model) == CNN) & (optimizer == optax.sgd):
        return reset_neurons_snr_CNN_parallel
    if (type(model) == MLP) & (optimizer == optax.adam):
        return reset_neurons_snr_adam_parallel
    if (type(model) == CNN) & (optimizer == optax.adam):
        return reset_neurons_snr_CNN_Adam_parallel