import jax.numpy as jnp
import jax.random as jr
from tensorflow.keras.datasets import mnist

def load_mnist():
    """
    Load the MNIST dataset and return normalized and flattened training and test sets as
    jax.numpy arrays.
    """
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten the images and normalize the pixel values
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)

    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)

    return (X_train, y_train), (X_test, y_test)

def sample_subset_mnist(X_train, y_train, N, key):
    """
    Sample N mnist training examples. 
    """

    num_rows, num_pixels = X_train.shape
    
    shuffled_indices = jr.permutation(key, num_rows)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    return (X_train[0:N], y_train[0:N])