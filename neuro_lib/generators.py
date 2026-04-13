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


def generate_ar_coupled(N, alpha=0.5, beta=0.5, gamma=0.4, lag=1, sigma=0.1):
    X = np.zeros(N)
    Y = np.zeros(N)
    ex = np.random.normal(0, sigma, N)
    ey = np.random.normal(0, sigma, N)
    
    for t in range(max(lag, 1), N):
        X[t] = alpha * X[t-1] + ex[t]
        # Y dipende dal suo passato E dal passato di X
        Y[t] = beta * Y[t-1] + gamma * X[t-lag] + ey[t]
        
    return X, Y


def generate_oscillatory_coupled(N, dt=0.01, coupling=0.5):
    t = np.linspace(0, N*dt, N)
    # Segnale X: una sinusoide semplice
    X = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, N)
    
    # Segnale Y: la sua fase dipende dal valore passato di X
    Y = np.zeros(N)
    phase_y = 0
    for i in range(1, N):
        # La variazione di fase di Y è influenzata da X[i-1]
        freq_y = 10 + coupling * X[i-1] 
        phase_y += 2 * np.pi * freq_y * dt
        Y[i] = np.sin(phase_y) + np.random.normal(0, 0.1, N)[i]
        
    return X, Y