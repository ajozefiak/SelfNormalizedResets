import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import struct
from typing import Any
from jax import lax
# Optax for Adam update
import optax

from .MLP import *

def init_fn_sgd(model, input_shape, seed, lr, threshold, reg_str = 0, algorithm = 0, reset_freq = 0, decay_rate = 0.99, threshold_reset_freq = 16, threshold_percentile = 0.99, threshold_expansion_factor = 2):
    rng = jr.PRNGKey(jnp.array(seed, int))
    dummy_input = jnp.ones((1, *input_shape))
    params = model.init(rng, dummy_input)['params']
    reg_params = set_reg_params(algorithm, params)
    lr, reg_str = set_lr_reg_str(algorithm, lr, reg_str)
    # Additional code for Adam update
    optimizer = optax.sgd(lr)
    opt_state = optimizer.init(params)

    return LearnerStateAdam(params=params, lr = lr, threshold = threshold, reg_str = reg_str,
                        reg_params = reg_params, algorithm = algorithm, opt_state = opt_state, reset_freq = reset_freq, decay_rate = decay_rate,
                        threshold_reset_freq = threshold_reset_freq, threshold_percentile = threshold_percentile, threshold_expansion_factor = threshold_expansion_factor)