import numpy as np


def binary_to_bipolar(v):
    """
    Convert a binary (0/1) vector to bipolar form (-1/1) before feeding it to the Hopfield network.

    Args:
        v (list or np.ndarray): Input binary vector

    Returns:
        np.ndarray: Bipolar vector
    """
    arr = np.array(v, dtype=int)
    return 2 * arr - 1


def train(patterns):
    """
    Train the Hopfield network on multiple patterns.

    Compute the outer product of each pattern (an N x N matrix of correlations),
    sum these correlation matrices, and normalize by N. Set the diagonal to 0 to remove self-feedback.

    Args:
        patterns (list of np.array): List of bipolar patterns

    Returns:
        np.ndarray: Weight matrix of the network (NxN)
    """
    N = len(patterns[0]) # Number of neurons
    W = np.zeros((N, N)) # Initialize weight matrix
    
    # Add outer product of pattern to weight matrix
    for p in patterns: 
        W += np.outer(p, p) # Hebbian rule
    
    # Zero the diagonal to prevent self-feedback
    np.fill_diagonal(W, 0)

    # Normalize by number of neurons
    W /= N
    return W


def recall(W, pattern, steps=5):
    """
    Recall a stored pattern from potentially noisy or partial input.

    Updates neurons asynchronously to push the state into the closest stored attractor.

    Args:
        W (np.ndarray): Weight matrix of the network
        pattern (np.ndarray): Initial state vector (bipolar)
        steps (int): Number of asynchronous updates

    Returns:
        np.ndarray: Recalled pattern (bipolar)
    """
    # Preserve original input
    state = pattern.copy()

    # Repeat asynchronous updates for fixed number of steps
    for _ in range(steps):
        # Random order of neuron indices to be updated
        indices = np.random.permutation(len(state))
        for i in indices:
            # Calculate weighted input to neuron i
            input_sum = np.dot(W[i], state)
            # Update neuron i based on threshold 
            state[i] = 1 if input_sum >= 0 else -1

    return state


