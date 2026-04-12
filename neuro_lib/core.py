from numba import njit
import numpy as np

"""
This file should contain all the functions decorated with numba (@njit)
and those starting with "_" (== functions that are not supposed to be called directly)
"""

@njit(cache=True)
def entropy_pmf_numba(p):
    #Computes Shannon entropy of a probability mass function (p.m.f.)
    #p: probability vector (entries sum up to 1)
    if abs(np.sum(p) - 1.0) > 1e-6:
        raise ValueError("PMF not normalized")
    entropy = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            entropy -= p[i] * np.log2(p[i])
    return entropy


# -----------------------------------------------
#               Binning Method 
# -----------------------------------------------

@njit(cache=True)
def _entropy_core(counts, bias_correction = None):
    """
    Estimates entropy of a probability mass function (pmf)
    from empirical frequencies
    counts: empirical observations, divided per category (bins, classes, whathever)
    """
    total = np.sum(counts)
    if total == 0:
        return 0.0
    res = 0.0
    m = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            m += 1
            p = counts[i] / total
            res -= p * np.log2(p)
    if bias_correction == "miller":
        res += (m - 1)/(2 * total)
    return res


@njit(cache=True)
def _bin_data_1d(x, n_bins, x_min, x_max):
    """Manual 1D histogramming."""
    counts = np.zeros(n_bins, dtype=np.int64)
    if x_max == x_min:
        return counts
    
    width = (x_max - x_min) / n_bins
    for i in range(len(x)):
        idx = int((x[i] - x_min) / width)
        if idx >= n_bins: idx = n_bins - 1
        elif idx < 0: idx = 0
        counts[idx] += 1
    return counts

@njit(cache=True)
def _entropy_absolute_njit(X, n_bins, bias_correction = None):
    counts = _bin_data_1d(X, n_bins, X.min(), X.max())
    return _entropy_core(counts, bias_correction)

@njit(cache=True)
def _entropy_conditional_njit(S, X, n_bins, bias_correction=None):
    """
    Compute conditional entropy H(X | S) using binning.

    Parameters
    ----------
    S : array (n_samples,)
        The conditioning variable. Must be the FIRST argument.
    X : array (n_samples,)
        The variable whose entropy is being computed. Must be the SECOND argument.
    n_bins : int
    bias_correction : str or None

    Returns
    -------
    float
        H(X | S)

    Note
    ----
    This function is NOT symmetric: _entropy_conditional_njit(S, X) != _entropy_conditional_njit(X, S).
    Always pass the conditioning variable first.
    """
    n_obs = len(S)
    s_min, s_max = S.min(), S.max()
    x_min, x_max = X.min(), X.max()

    s_width = (s_max - s_min) / n_bins if s_max > s_min else 1.0
    x_width = (x_max - x_min) / n_bins if x_max > x_min else 1.0

    counts_2d = np.zeros((n_bins, n_bins), dtype=np.int64)
    s_bin_totals = np.zeros(n_bins, dtype=np.int64)

    for j in range(n_obs):
        s_idx = int((S[j] - s_min) / s_width)
        if s_idx >= n_bins: s_idx = n_bins - 1
        elif s_idx < 0: s_idx = 0

        x_idx = int((X[j] - x_min) / x_width)
        if x_idx >= n_bins: x_idx = n_bins - 1
        elif x_idx < 0: x_idx = 0

        counts_2d[s_idx, x_idx] += 1
        s_bin_totals[s_idx] += 1

    h_cond = 0.0
    for i in range(n_bins):
        subset_size = s_bin_totals[i]
        if subset_size > 0:
            p_s = subset_size / n_obs
            h_cond += p_s * _entropy_core(counts_2d[i, :], bias_correction)

    return h_cond

def _entropy_binning_1d_numba(data, bin_number, which="absolute", bias_correction = None):
    """
    Python wrapper to handle the 'which' logic and dispatch to Numba.
    """
    S = data[:, 0]
    X = data[:, 1]
    
    if which == "absolute":
        return _entropy_absolute_njit(X, bin_number, bias_correction)
    elif which == "conditional":
        return _entropy_conditional_njit(S, X, bin_number, bias_correction)
    else:
        raise ValueError("Invalid 'which' parameter. Use 'absolute' or 'conditional'.")


def mi_binning_2d_numba(data, bin_number, bias_correction = None):
    """
    Discretizes continuous data and computes the mutual information of the corresponding pmf
    """
    S = data[:, 0]
    X = data[:, 1]
    Hx = _entropy_binning_1d_numba(data, bin_number, "absolute" ,bias_correction)
    Hx_given_s = _entropy_binning_1d_numba(data, bin_number, "conditional", bias_correction)
    return Hx - Hx_given_s


@njit(cache=True)
def histogram_error_numba(x, bin_number):
    """
    Computes average relative error over the bin counts
    In the Poisson limit, 
    err = sqrt(counts)
    rel.err = sqrt(counts)/counts
    """
    n_obs = len(x)
    if n_obs == 0:
        return 0.0
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return 1.0 / np.sqrt(n_obs)
        
    counts = np.zeros(bin_number, dtype=np.int64)
    width = (x_max - x_min) / bin_number
    
    for i in range(n_obs):
        idx = int((x[i] - x_min) / width)
        if idx >= bin_number:
            idx = bin_number - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1

    total_error = 0.0
    active_bins = 0
    
    for i in range(bin_number):
        c = counts[i]
        if c > 0:
            total_error += 1.0 / np.sqrt(c)
            active_bins += 1
    if active_bins == 0:
        return 0.0   
    return total_error / active_bins

# -----------------------------------------------
#       Joint Gaussian -- Cond. Variance
# -----------------------------------------------
def _conditional_variance(X, Y):
    """
    Compute Var(X | Y) where:
    X: shape (n,)
    Y: shape (n,) or (n, k)
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Build joint vector [X, Y]
    XY = np.column_stack((X, Y))

    cov = np.cov(XY, rowvar=False)

    var_X = cov[0, 0]
    cov_XY = cov[0, 1:]
    cov_YY = cov[1:, 1:]

    # Solve instead of inverse (more stable)
    inv_term = np.linalg.solve(cov_YY, cov_XY)

    cond_var = var_X - cov_XY @ inv_term
    return cond_var