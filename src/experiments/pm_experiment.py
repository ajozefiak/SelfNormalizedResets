# TODO: do we need pickle
import pickle
import jax.random as jr
import numpy as np
# For saving results to Github
import os

from SelfNormalizedResets.src.algorithms import *
from SelfNormalizedResets.src.data.sample_data import *

def save_to_disk(data, file_name):
    # Extract the directory portion of file_name
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.exists(dir_name):
        # Create the directory (and any parents) if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)
        
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def run_snr_permuted_mnist_experiment(num_tasks, epochs, batch_size, seed, task_seed,
                                 states, optimizer, model,
                                 X_train, y_train, X_test, y_test, num_classes,
                                 file_name, threshold_file_name, save_freq, verbose, tau_max):

    # Get the directory where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "../../results/pm_mnist/")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Update paths for logging results
    file_name = os.path.join(save_dir, file_name)
    threshold_file_name = os.path.join(save_dir, threshold_file_name)

    # Get function_dictionary with training functions
    training_function_dictionary = training_factory(model, num_classes, optimizer)
    
    # Extract Training functions
    accuracy_parallel = training_function_dictionary['accuracy_parallel']
    train_step_optax_parallel = training_function_dictionary['train_step_optax_parallel']

    # Get reset_snr_parallel based on model architecture and optimizer
    reset_snr_parallel = get_reset_snr_parallel(model, optimizer)
    
    # Initialize neuron ages, need to call get_neurons once
    neurons_parallel = get_neurons_parallel(states, X_train[0:1])
    neuron_ages_parallel = initialize_neuron_ages_batch_parallel(neurons_parallel)
    neuron_ages_hist_parallel = initialize_neuron_ages_hist_parallel(neuron_ages_parallel, tau_max)
    # Reshape Tau from a single float to a dictionary of layers => floats so that each neuron has its own threshold
    states = reinitialize_tau_indiv_parallel(states, neuron_ages_parallel)

    # Initialize key and task_key from seed and task_seed
    key = jr.PRNGKey(seed)
    task_key = jr.PRNGKey(task_seed)

    # Initialize data structures for storing accuracies
    N = len(states.algorithm)
    task_len = epochs * int(len(X_train) / batch_size)
    T = task_len*num_tasks
    batch_accuracies = np.zeros((N,T))
    avg_task_accuracies = np.zeros((N, num_tasks))

    # Indexing for batch_accuracies and avg_task_accuracies
    T_acc = 0

    for task in range(num_tasks):
        task_key, task_split_key = jr.split(task_key)
        X_train, y_train, X_test, y_test = pm_next_task(task_split_key, X_train, y_train, X_test, y_test)
        for i in range(0, len(X_train), batch_size):
            key, rand_key = jr.split(key)

            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Get neuron firing for this batch and increment dead neurons
            neurons_parallel = get_neurons_parallel(states, x_batch)
            neuron_ages_parallel, neuron_ages_hist_parallel = increment_neuron_ages_snr_parallel(neurons_parallel, neuron_ages_parallel, neuron_ages_hist_parallel)

            acc = accuracy_parallel(states, x_batch, y_batch)

            # update batch_accuracies
            batch_accuracies[:,T_acc] = np.array(acc)
            T_acc += 1
                
            # Perform SGD/Adam update, possibly with L2 regularization contingent on algorithm
            states = train_step_optax_parallel(states, (x_batch, y_batch))

            # Neuron Reset Step.
            states, neuron_ages_parallel, neuron_ages_hist_parallel = reset_snr_parallel(states, neuron_ages_parallel, neuron_ages_hist_parallel, rand_key, X_train[0])

        # Update the neurons' reset-thresholds tau contingent on eta, frequency of updates, and historical firing rates
        states, neuron_ages_hist_parallel = update_tau_parallel(states, neuron_ages_hist_parallel, task+1)        

        total_avg_accuracy = np.mean(batch_accuracies[:,:T_acc], axis=1)
        avg_task_accuracy = np.mean(batch_accuracies[:, T_acc - task_len: T_acc] , axis=1)
        avg_task_accuracies[:,task] = avg_task_accuracy

        # If verbose, print average accuracy and total average accuracy
        if verbose:
            print(f"Average Accuracy for Task {task+1}: {avg_task_accuracy}")
            print(f"Average Accuracy for all tasks up to Task {task+1}: {total_avg_accuracy}")

        # Save avg_task_accuracies
        if task % save_freq == 0:
            save_to_disk(avg_task_accuracies, file_name)

            threshold_data = {"task": task,
                                "threshold": states.threshold,
                                "ages": neuron_ages_parallel} 

            save_to_disk(threshold_data, threshold_file_name + f"_task_{task}.pkl")


    # Save final avg_task_accuracies
    save_to_disk(avg_task_accuracies, file_name)
    threshold_data = {"task": num_tasks,
                        "threshold": states.threshold,
                        "ages": neuron_ages_parallel} 

    save_to_disk(threshold_data, threshold_file_name + f"_task_{num_tasks}.pkl")