import numpy as np



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