import jax.random as jr

def pm_next_task(key, X_train, y_train, X_test, y_test):
    """
    Sample the next Permuted MNIST task by shuffling the order of images and permuting pixels.
    """
    
    num_rows, num_pixels = X_train.shape

    key, split_key = jr.split(key)
    shuffled_indices = jr.permutation(split_key, num_rows)
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    key, split_key = jr.split(key)
    pixel_permutation = jr.permutation(split_key, num_pixels)
    X_train = X_train[:, pixel_permutation]
    X_test = X_test[:, pixel_permutation]
    return X_train, y_train, X_test, y_test