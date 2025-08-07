import sys
import math

import tiktoken
import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
import os
import numpy as np

def get_dataloader(text):

    def process_text(text):
        text = text.replace('\n', ' ')
        return text

    text = process_text(text)

    enc = tiktoken.get_encoding("gpt2")

    # Encode text as tokens, as a list and jax.numpy array
    tokens_list = enc.encode(text)
    tokens_jnp = jnp.array(tokens_list)

    # Get the unique tokens and their indices to build a map
    unique_tokens = jnp.unique(tokens_jnp)
    unique_tokens, unique_tokens_idx = jnp.unique(unique_tokens, return_index = True)

    # Create the dictionary mapping unique tokens to their indices
    token_to_index_dict = {int(unique_tokens[i]): int(unique_tokens_idx[i]) for i in range(len(unique_tokens))}
    index_to_token_dict = {int(unique_tokens_idx[i]): int(unique_tokens[i]) for i in range(len(unique_tokens))}

    def custom_encoder(x):
        return token_to_index_dict.get(x, None)  # Returns None if x is not in the dictionary

    def custom_decoder(x):
        return index_to_token_dict.get(x, None)  # Returns None if x is not in the dictionary

    def custom_token_list_encoding(tokens):
        tokens_new = []
        for i in range(len(tokens)):
            tokens_new.append(custom_encoder(tokens[i]))
        return tokens_new

    def custom_token_list_decoder(tokens):
        tokens_new = []
        for i in range(len(tokens)):
            tokens_new.append(custom_decoder(tokens[i]))
        return tokens_new

    num_unique_tokens = len(unique_tokens)
    print(f"Number of unique tokens: {num_unique_tokens}")

    def permute_text_vocabulary(text, key):
        # Split the text into a list of words or characters
        words = text.split()

        # Get the unique vocabulary
        unique_words = list(set(words))

        # Generate a random permutation of indices
        permuted_indices = jr.permutation(key, jnp.arange(len(unique_words)))

        # Create a mapping from original words to permuted counterparts using the permuted indices
        word_map = {original: unique_words[idx] for original, idx in zip(unique_words, permuted_indices)}

        # Map the original text to the permuted vocabulary
        permuted_text = ' '.join(word_map[word] for word in words)

        return permuted_text

    def permute_rows(tokens, T, key):
        # Randomize the order of examples
        num_rows = tokens.shape[0] // T
        tokens = tokens[:num_rows * T] # Trim excess tokens if any

        # Step 1: Reshape into contiguous batches
        rows = tokens.reshape((num_rows, T))

        # Step 2: Shuffle the batches
        # Generate a permutation of indices for the number of batches
        permuted_indices = jax.random.permutation(key, num_rows)

        # Apply permutation to shuffle batches
        shuffled_rows = rows[permuted_indices]

        # Step 3: Flatten back to 1D array if necessary
        shuffled_tokens = shuffled_rows.flatten()

        return shuffled_tokens
    
    # B: batch size
    # T: Context window
    # N: Number of Tokens per Task
    class DataLoaderPermuteText:
        def __init__(self, text, B, T, N, key):
            self.current_position = 0
            self.B = B
            self.T = T
            self.N = N
            self.key = key

            enc = tiktoken.get_encoding("gpt2")

            text = process_text(text)
            text = ' ' + permute_text_vocabulary(text, key)

            tokens_list = enc.encode(text)[0:N]
            
            # limited vocab tokens
            tokens_lv = custom_token_list_encoding(tokens_list)

            self.tokens = jnp.array(tokens_lv)
            print(f"loaded {len(self.tokens)} tokens in the datasets" )
            print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

        def next_batch(self):
            B,T = self.B, self.T

            buf = self.tokens[self.current_position:self.current_position+B*T]
            x,y = jnp.reshape(buf, (B,T))[:,:-1], jnp.reshape(buf, (B,T))[:,1:]

            self.current_position += B*T
            if self.current_position + B*T > len(self.tokens):
                self.current_position = 0
                
                # Here, we permute the examples
                key, split_key = jr.split(self.key)
                self.key = key
                self.tokens = permute_rows(self.tokens, T, split_key)
            return x,y  
    
    return DataLoaderPermuteText   

# Note need to pass algorithm/model data/config when refactoring of experiment is finished
def run_experiment_PS(text, B, T, N, epochs, tasks, seed):
    
    # Get the data_loader_class
    data_loader_class = get_dataloader(text)

    # Initialize the random key
    random_key = jr.PRNGKey(seed)
    task_random_key = jr.PRNGKey(seed)
    print(f"Initial key: {random_key}, Type: {type(random_key)}")
    random_key, split_key = jr.split(random_key)

    # Get the total number of training steps
    train_steps_per_task = (N * epochs) // (B * T)

    
    # TODO HERE: NETWORK MUST BE INITIALIZED AT THIS POINT

    # Iniitialize loss_array
    loss_array = np.zeros(train_steps_per_task * tasks)
    t = 0
   
    for task in range(tasks):
    
    
        print(f"Task: {task}")

        # Split the random key
        task_random_key, task_split_key = jr.split(task_random_key)
        data_loader = data_loader_class(text=text, B=B, T=T, N=N, key=task_split_key)
        
        for step in range(train_steps_per_task):
            
            # LOAD BATCH OF DATA
            x,y = data_loader.next_batch()
            
            # TODO HERE: 
            # (1) PERFORM A FORWARD PASS ON BATCH (X,Y) AND GET LOSS
            loss = 0.0
            # (2) RECORD NEURON ACTIVITIES AND OTHER COVARIATES
            # (3) PERFORM BACKPROP AND ANY NEURON RESETS OR OTHER CONTINUAL LEARNING INTERVENTIONS

            
            # Update loss_array (and any other logging of covariates)
            loss_array[t] = loss
            t += 1

    return loss_array



# Experiment parameters
seed = 0
# In our paper we tested scales = 1, 4, and 16
scale = 1
epochs = 100
tasks = 500
# batch size
B = 8 
# Context window T = 128 + 1, to account for a buffer so as to shuffle (x,y) pairs correctly
T = 128+1

# Load all_shakespeare.txt
# TODO: make sure that your path to all_shakespeare.txt is correct
# dir_path = ...
path = os.path.join(dir_path, 'all_shakespeare.txt')

with open(path, 'r') as f:
  text = f.read()

# Scale params
# We increase model scale by doubling width and hence squaring the number of weights
width_factor = int(round(math.sqrt(scale)))

n_head = 2 * width_factor
n_embd = 32 * width_factor
n_neurons = 256 * width_factor

batches = int(32 * (scale ** (0.74)))
N = int(B * T * batches)

# TODO: add back model parameters/config to the function call
run_experiment_PS(text, B, T, N, epochs, tasks, seed)
