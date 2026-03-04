"""Utility functions for common operations"""

import math
import random
import numpy as np

def compute_path_objectives(path, G, weights):
    """
    Compute the weighted objectives for a given path
    """
    num_objectives = len(weights)
    cum__weighted_objectives = [0.0] * len(weights)

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_objectives = G.edges[u, v]['objectives']

        for j in range(num_objectives):
            cum__weighted_objectives[j] += weights[j] * edge_objectives[j]
    
    return cum__weighted_objectives


def sample_log_scale(min_val, max_val):
    """
    Sample log-uniform between min and max provided
    """  
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    
    log_sample = random.uniform(log_min, log_max)
    sample = 10 ** log_sample
    return sample


def normalize_weights(weights):
    """
    # Sum to 1 normalization
    """
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights must not be zero for normalization.")
    return [w / total for w in weights]


def normalize_solutions(*vecs):
    """
    Normalize vectors per min max normalization
    """
    M = np.asarray(vecs, dtype=float)  # shape (n_vectors, n_obj)

    mins = np.min(M, axis=0)  
    maxs = np.max(M, axis=0)  
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  

    M_norm = (M - mins) / ranges
    return [row.tolist() for row in M_norm]


def compute_optimality_error(wm_objectives, sol_objectives):
    """
    Compute the average percentage that each objective is off by
    We take the percentage error of the all objectives that perform worse and then take the average percentage error
    """
    if len(wm_objectives) != len(sol_objectives):
        raise ValueError("Vectors must have the same length.")

    total_pct = 0.0
    for wm_val, sol_val in zip(wm_objectives, sol_objectives):
        if sol_val > wm_val:
            overshoot = sol_val - wm_val         
            total_pct += (overshoot / (wm_val + 1e-5)) * 100.0  

    return total_pct

def mean_absolute_deviation(vector):
    """
    Compute MAD of provided vector
    """
    n = len(vector)
    mu = sum(vector)/n
    return sum(abs(c - mu) for c in vector)/n
