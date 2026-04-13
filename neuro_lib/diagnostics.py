import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.spatial.distance import cdist
from .estimators import estimate_mi

def check_stationarity(data, significance=0.05):
    """
    Perform Augmented Dickey-Fuller test to check stationarity.
    
    Returns:
    -------
    is_stationary : bool
    p_value : float
    """
    res = adfuller(data)
    p_value = res[1]
    return p_value < significance, p_value

def find_optimal_delay(signal, max_lag=50, method='gc'):
    """
    Find optimal embedding delay (tau) using the first local minimum 
    of Average Mutual Information (AMI).
    """
    mi_values = []
    lags = np.arange(1, max_lag + 1)
    
    for lag in lags:
        s_t = signal[lag:]
        s_lag = signal[:-lag]
        data = np.column_stack((s_t, s_lag))
        
        mi = estimate_mi(data, method=method)
        mi_values.append(mi)
    
    mi_values = np.array(mi_values)
    
    # Find first local minimum 
    minima = (mi_values[1:-1] < mi_values[:-2]) & (mi_values[1:-1] < mi_values[2:])
    indices = np.where(minima)[0]
    
    # Se non trova minimi, restituisce il lag con MI minima nel range
    tau_opt = lags[indices[0] + 1] if len(indices) > 0 else lags[np.argmin(mi_values)]
    
    return tau_opt, lags, mi_values

def find_optimal_dimension(signal, tau, max_m=10, R_tol=15.0):
    """
    Find optimal embedding dimension (m) using False Nearest Neighbors (FNN).
    """
    def embed(sig, dim, delay):
        n_vec = len(sig) - (dim - 1) * delay
        return np.array([sig[i : i + dim * delay : delay] for i in range(n_vec)])

    fnn_percentages = []
    m_range = np.arange(1, max_m)
    
    for m in m_range:
        Y_m = embed(signal, m, tau)
        Y_m1 = embed(signal, m + 1, tau)
        
        n_common = len(Y_m1)
        Y_m = Y_m[:n_common]
        
        # Distanze nel piano a dimensione m
        # Usiamo un subset per velocizzare se il segnale è molto lungo
        step = max(1, n_common // 1000) 
        Y_m_sub = Y_m[::step]
        
        dists = cdist(Y_m_sub, Y_m_sub)
        np.fill_diagonal(dists, np.inf)
        nn_indices = np.argmin(dists, axis=1)
        
        # Distanza nel piano m+1
        # La differenza è data solo dalla nuova coordinata aggiunta
        r_m = dists[np.arange(len(Y_m_sub)), nn_indices]
        r_m1 = np.abs(Y_m1[::step][:, -1] - Y_m1[nn_indices * step][:, -1])
        
        # Criterio di Falsità
        # Se r_m1 / r_m > R_tol, il vicino era "falso"
        is_fnn = (r_m1 / r_m > R_tol)
        fnn_percentages.append(np.mean(is_fnn) * 100)
        
    fnn_percentages = np.array(fnn_percentages)
    # m_opt è il primo valore dove FNN < 1%
    opt_idx = np.where(fnn_percentages < 1.0)[0]
    m_opt = m_range[opt_idx[0]] if len(opt_idx) > 0 else max_m
    
    return m_opt, m_range, fnn_percentages