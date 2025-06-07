# TODO: do we need to import jax?
import jax
from jax import jit, grad, vmap
import jax.numpy as jnp
from flax import linen as nn
# Optax for Adam update
import optax

# Factory function that returns a dict of JITed and vmapped training functions 
# that close over model, num_classes, and optimizer from the outer scope.
# NOTE: Every agent in a vectorized LearnerState must use the same model, num_classes, and optimizer 
# (along with optimizer hyperparameters, except for learning rate as is currently implemented)
# NOTE: At the moment, each call to train_step_optax initializes a new solver using optimizer
# This allows each agent to potentialy have a different learning rate (for hyperparameter sweeps).
def training_factory(model, num_classes, optimizer):

        # @jax.jit
        def one_hot(x, k, dtype=jnp.float32):
            return jnp.array(x[:, None] == jnp.arange(k), dtype)

        @jax.jit
        def l2_regularization(params, reg_params, l2_reg_strength):
            l2_loss = 0.0
            for p, reg_p in zip(jax.tree_leaves(params), jax.tree_leaves(reg_params)):
                l2_loss += jnp.sum(jnp.square(p - reg_p))
            return l2_reg_strength * l2_loss

        @jax.jit
        def loss_fn(params, x_batch, y_batch, reg_params, l2_reg_strength):
            logits = model.apply({'params': params}, x_batch)
            cross_entropy_loss = -jnp.mean(jnp.sum(one_hot(y_batch, num_classes) * nn.log_softmax(logits), axis=-1))
            l2_loss = l2_regularization(params, reg_params, l2_reg_strength) 
            return cross_entropy_loss + l2_loss

        @jax.jit
        def accuracy(state, x_batch, y_batch):
            # TODO: is the line below redundant
            target_class = jnp.argmax(one_hot(y_batch, num_classes), axis=1)
            predicted_class = jnp.argmax(model.apply({'params': state.params}, x_batch), axis=1)
            return jnp.mean(predicted_class == target_class)


        accuracy_parallel = jax.jit(vmap(accuracy, in_axes=(0,None,None)))

        compute_grads = jax.jit(grad(loss_fn))

        @jit
        def sgd_update(params, lr, grads):
            return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

        # TODO: Change @jax.jit to @jit everywhere here, I should be consistent
        @jax.jit
        def train_step(state, batch):
            # Load parameters of LearnerState
            params = state.params
            lr = state.lr
            reg_str = state.reg_str
            reg_params = state.reg_params
            
            grads = compute_grads(params, batch[0], batch[1], reg_params, reg_str)
            params = sgd_update(params, lr, grads)

            return state.replace(params=params)

        @jax.jit
        def train_step_optax(state, batch):
            # Load parameters of LearnerState
            params = state.params
            lr = state.lr
            reg_str = state.reg_str
            reg_params = state.reg_params
            opt_state = state.opt_state

            grads = compute_grads(params, batch[0], batch[1], reg_params, reg_str)
            # Optax Update
            solver = optimizer(lr)
            updates, opt_state = solver.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            return state.replace(params=params, opt_state = opt_state)

        train_step_parallel = jax.jit(vmap(train_step, in_axes=(0, None)))

        train_step_optax_parallel = jax.jit(vmap(train_step_optax, in_axes=(0, None)))

        function_dictionary = {
            "accuracy": accuracy,
            "accuracy_parallel": accuracy_parallel,
            "train_step": train_step,
            "train_step_parallel": train_step_parallel,
            "train_step_optax": train_step_optax,
            "train_step_optax_parallel": train_step_optax_parallel
        }

        return function_dictionary