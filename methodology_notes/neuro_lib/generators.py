import numpy as np

"""
This file should contain all the functions that generate the stimuli and neural response variables
"""

# =============================================================
#                      Data Generators
# =============================================================

def generate_neural_variable(S, f, noise_std=1.0, tau=0, phi=0.0):
    """
    Unified neural variable generator.
    X[t] = phi * X[t-1] + f(S[t - tau]) + noise_std * eta[t]
    """
    Ntsteps = len(S)
    eta = np.random.normal(0, noise_std, Ntsteps)
    X = np.zeros(Ntsteps)
    
    S_shifted = np.roll(S, tau)
    if tau > 0:
        S_shifted[:tau] = np.nan # Or a default value
        
    driven_component = f(S_shifted)

    if phi == 0:
        return driven_component + eta
    
    for t in range(1, Ntsteps):
        if np.isnan(driven_component[t]): continue
        X[t] = phi * X[t-1] + driven_component[t] + eta[t]
        
    return X


def generate_data(N, sigma, f= lambda s:s):
    """
        X = f(S) + sigma * eta
        S: normal random variable of std = 1, mean = 0
    """
    S = np.random.normal(0, 1, N)
    X = f(S) + np.random.normal(0, sigma, N)
    return np.column_stack((S, X))

# =============================================================
#          Data Generators for transfer entropy analysis
# =============================================================


def generate_ar_coupled(N, alpha=0.7, beta=0.3, gamma=0.5, lag=5, noise_std=0.1):
    """
    Generates two AR(1) coupled processes
    X(t) = alpha * X(t-1) + noise
    Y(t) = beta * Y(t-1) + gamma * X(t-lag) + noise
    """
    X = np.zeros(N)
    Y = np.zeros(N)
    
    # initialization with white noise
    noise_x = np.random.normal(0, noise_std, N)
    noise_y = np.random.normal(0, noise_std, N)
    
    for t in range(max(lag, 1), N):
        X[t] = alpha * X[t-1] + noise_x[t]
        # Y depends on its own past and X's past
        Y[t] = beta * Y[t-1] + gamma * X[t-lag] + noise_y[t]
        
    return X, Y

def generate_oscillatory_coupled(N, dt=0.05, coupling=0.4, lag=10, noise_std=0.05):
    """
    neuro-inspired model: X and Y are two coupled oscillators
    """
    t = np.arange(N) * dt
    X = np.sin(2 * np.pi * 2 * t) 
    
    # add noise to X
    X += np.random.normal(0, noise_std, N)
    
    Y = np.zeros(N)
    phase_y = 0
    # coupling lag
    X_shifted = np.roll(X, lag)
    
    for i in range(1, N):
        # freqY = freq_base + coupling * past_source
        freq_y = 5 + coupling * X_shifted[i] 
        phase_y += 2 * np.pi * freq_y * dt
        Y[i] = np.sin(phase_y)
        
    # add noise to Y
    Y += np.random.normal(0, noise_std, N)
    
    return X, Y