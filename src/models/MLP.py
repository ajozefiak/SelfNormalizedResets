import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import struct
from typing import Any
from jax import lax
# Optax for Adam update
import optax

class MLP(nn.Module):
    num_classes: int
    width: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

class CNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # First convolutional layer
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        # Second convolutional layer
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        # Flatten before passing to the fully connected layers
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        
        # First fully connected layer
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)

        # Second fully connected layer
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)

        # Output connected layer
        x = nn.Dense(features=self.num_classes)(x)

        return x
        

# Algorithm 0 == L2 Init
# Algorithm 1 == L2 Regulaization
# Algorithm 2 == Shrink and Perturb
# Algorithm 3 == Our Neuron Reset
# Algorithm 4 == ReDO (Task Boundary Agnostic)
# Algorithm 5 == EWC (Task Boundary Agnostic)
# Algorith 33 == Our Neuron Reset with L2 regularization

@struct.dataclass
class LearnerState:
    params: Any
    lr: Any
    threshold: Any
    reg_str: Any
    reg_params: Any 
    algorithm: Any
    reset_freq: Any
    decay_rate: Any
    # TODO: merge threshold_reset_freq and reset_freq into a single variable
    threshold_reset_freq: Any
    threshold_percentile: Any
    threshold_expansion_factor: Any

def zero_params(params):
    return jax.tree_map(lambda x: jnp.zeros_like(x), params)

def set_reg_params(algorithm, params):
    # Set reg_params = params if using Algorithm L2-init
    # TODO: This first line is redundant, but I leave it in for now as it
    # may be explanatory for future purposes, as I add more parameters
    # Algorithm 0 == L2 Init
    reg_params = lax.cond(algorithm == 0, lambda x: x, lambda x: x, params)
    # Algorithm 1 == L2 Regularization
    # Algorithm 2 == Shrink and Perturb
    # Algorithm 33 == Resets with L2
    reg_params = lax.cond((algorithm == 1) | (algorithm == 2) | (algorithm == 33), zero_params, lambda x: x, reg_params)
    # TODO: Implement other choices of algorithms
    return reg_params

def set_lr_reg_str(algorithm, lr, reg_str):
    # Algorithm 2 == Shrink and Perturb
    # Due to shrink-age: params = reg_str * (params - lr * grad) + threshold * rand_params
    # new_lr = reg_str * lr
    # new_reg_str = 0.5 * (1 - reg_str) / (new_lr)
    # where reg_str is the shrkinage parameter, threshold is noise-scale parameter
    # TODO: Perhaps add these as separate variables to LearnerState, such a shrink and sigma

    lr, reg_str = lax.cond(algorithm == 2, lambda lr_, reg_str_: (reg_str_ * lr_, 0.5 * (1 - reg_str_) / (reg_str_ * lr_)), lambda lr_, reg_str_: (lr_, reg_str_), lr, reg_str)
    return (lr, reg_str)

def init_fn(model, input_shape, seed, lr, threshold, reg_str = 0, algorithm = 0, reset_freq = 0, decay_rate = 0.99, threshold_reset_freq = 16, threshold_percentile = 0.99, threshold_expansion_factor = 2):
    rng = jr.PRNGKey(jnp.array(seed, int))
    dummy_input = jnp.ones((1, *input_shape))
    params = model.init(rng, dummy_input)['params']
    reg_params = set_reg_params(algorithm, params)
    lr, reg_str = set_lr_reg_str(algorithm, lr, reg_str)
    return LearnerState(params=params, lr = lr, threshold = threshold, reg_str = reg_str,
                        reg_params = reg_params, algorithm = algorithm, reset_freq = reset_freq, decay_rate = decay_rate,
                        threshold_reset_freq = threshold_reset_freq, threshold_percentile = threshold_percentile, threshold_expansion_factor = threshold_expansion_factor)


##########################################################################################
###### ADAM UPDATE                                                                  ######
##########################################################################################

@struct.dataclass
class LearnerStateAdam:
    params: Any
    lr: Any
    threshold: Any
    reg_str: Any
    reg_params: Any 
    algorithm: Any
    opt_state: Any
    reset_freq: Any
    decay_rate: Any
    # TODO: merge threshold_reset_freq and reset_freq into a single variable
    threshold_reset_freq: Any
    threshold_percentile: Any
    threshold_expansion_factor: Any

def init_fn_adam(model, input_shape, seed, lr, threshold, reg_str = 0, algorithm = 0, reset_freq = 0, decay_rate = 0.99, threshold_reset_freq = 16, threshold_percentile = 0.99, threshold_expansion_factor = 2):
    rng = jr.PRNGKey(jnp.array(seed, int))
    dummy_input = jnp.ones((1, *input_shape))
    params = model.init(rng, dummy_input)['params']
    reg_params = set_reg_params(algorithm, params)
    lr, reg_str = set_lr_reg_str(algorithm, lr, reg_str)
    # Additional code for Adam update
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    return LearnerStateAdam(params=params, lr = lr, threshold = threshold, reg_str = reg_str,
                        reg_params = reg_params, algorithm = algorithm, opt_state = opt_state, reset_freq = reset_freq, decay_rate = decay_rate,
                        threshold_reset_freq = threshold_reset_freq, threshold_percentile = threshold_percentile, threshold_expansion_factor = threshold_expansion_factor)