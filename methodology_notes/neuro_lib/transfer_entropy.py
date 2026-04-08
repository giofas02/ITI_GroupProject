import numpy as np
from numba import njit
from .core import _entropy_core, _entropy_conditional_njit, _conditional_variance

# =============================================================
# 2D Conditional Entropy
# =============================================================
@njit(cache=True)
def _entropy_conditional_2d_njit(S, X, n_bins, bias_correction=None):
    """
    Compute conditional entropy H(X | S1, S2) using binning.

    Parameters
    ----------
    S : array (n_samples, 2)
        Two conditioning variables (e.g., source lag and target past)
    X : array (n_samples,)
        Variable whose entropy is computed

    Returns
    -------
    h_cond : float
        Conditional entropy H(X | S1, S2)
    """

    n_obs = len(X)

    # Split S into its two components
    s1, s2 = S[:, 0], S[:, 1]

    # Min/max for binning
    s1_min, s1_max = s1.min(), s1.max()
    s2_min, s2_max = s2.min(), s2.max()
    x_min, x_max = X.min(), X.max()

    # Bin widths
    s1_width = (s1_max - s1_min) / n_bins if s1_max > s1_min else 1.0
    s2_width = (s2_max - s2_min) / n_bins if s2_max > s2_min else 1.0
    x_width = (x_max - x_min) / n_bins if x_max > x_min else 1.0

    # 3D histogram: (S1, S2, X)
    counts_3d = np.zeros((n_bins, n_bins, n_bins), dtype=np.int64)

    # Counts for joint bins (S1, S2)
    s_bin_totals = np.zeros((n_bins, n_bins), dtype=np.int64)

    # Fill histogram
    for j in range(n_obs):

        idx1 = int((s1[j] - s1_min) / s1_width)
        idx2 = int((s2[j] - s2_min) / s2_width)
        x_idx = int((X[j] - x_min) / x_width)

        # Clip indices
        idx1 = min(max(idx1, 0), n_bins - 1)
        idx2 = min(max(idx2, 0), n_bins - 1)
        x_idx = min(max(x_idx, 0), n_bins - 1)

        counts_3d[idx1, idx2, x_idx] += 1
        s_bin_totals[idx1, idx2] += 1

    # Compute conditional entropy
    h_cond = 0.0

    for i in range(n_bins):
        for j in range(n_bins):
            subset_size = s_bin_totals[i, j]

            if subset_size > 0:
                p_s = subset_size / n_obs

                # Entropy of X restricted to (S1=i, S2=j)
                h_cond += p_s * _entropy_core(counts_3d[i, j, :], bias_correction)

    return h_cond


# =============================================================
# Transfer Entropy Binning
# =============================================================
def transfer_entropy_binning(source, target, n_bins=10, lag=1, bias_correction=None):
    """
    Compute Transfer Entropy TE(source -> target).

    TE = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, S_{t-lag})

    Parameters
    ----------
    source : array
    target : array
    lag : int
        Lag applied to source

    Returns
    -------
    TE : float
    """

    # Lagged source (S_{t-lag})
    S_lag = np.roll(source, lag)
    S_lag[:lag] = np.nan  # invalid entries

    # Past of target (X_{t-1})
    X_past = np.roll(target, 1)
    X_past[0] = np.nan

    # Current target (X_t)
    X_t = target

    # Remove invalid points
    mask = ~np.isnan(S_lag) & ~np.isnan(X_past)

    S_lag = S_lag[mask]
    X_past = X_past[mask]
    X_t = X_t[mask]

    # First term: H(X_t | X_{t-1})
    h1 = _entropy_conditional_njit(X_past, X_t, n_bins, bias_correction)

    # Second term: H(X_t | X_{t-1}, S_{t-lag})
    S_joint = np.column_stack((S_lag, X_past))
    h2 = _entropy_conditional_2d_njit(S_joint, X_t, n_bins, bias_correction)

    # Transfer Entropy
    TE = h1 - h2

    return TE


# =============================================================
# Transfer Entropy Gaussian 
# =============================================================
def transfer_entropy_gaussian(source, target, lag=1):
    """
    Gaussian Transfer Entropy TE(source -> target)
    """

    # Build lagged variables
    S_lag = np.roll(source, lag)
    S_lag[:lag] = np.nan

    X_past = np.roll(target, 1)
    X_past[0] = np.nan

    X_t = target

    # Remove NaNs
    mask = ~np.isnan(S_lag) & ~np.isnan(X_past)
    S_lag = S_lag[mask]
    X_past = X_past[mask]
    X_t = X_t[mask]

    # Var(X_t | X_past)
    var1 = _conditional_variance(X_t, X_past)

    # Var(X_t | X_past, S_lag)
    Y_joint = np.column_stack((X_past, S_lag))
    var2 = _conditional_variance(X_t, Y_joint)

    # TE
    TE = 0.5 * np.log(var1 / var2)

    return TE


# =============================================================
# Transfer Entropy Computation Interface
# =============================================================
def transfer_entropy(source, target, method="binning", **kwargs):
    """
    General TE interface.

    Parameters
    ----------
    method : str
        "binning" or "gaussian"
    kwargs :
        passed to the chosen method
    """

    if method == "binning":
        return transfer_entropy_binning(source, target, **kwargs)

    elif method == "gaussian":
        return transfer_entropy_gaussian(source, target, **kwargs)

    else:
        raise ValueError("Method must be 'binning' or 'gaussian'")


# =============================================================
# Transfer Entropy Matrix
# =============================================================
def transfer_entropy_matrix(data_matrix, method="binning", **kwargs):
    """
    Compute full Transfer Entropy matrix.

    Parameters
    ----------
    data_matrix : array (n_regions, n_timepoints)
    method : str
        "binning" or "gaussian"
    kwargs :
        passed to TE estimator

    Returns
    -------
    TE_mat : array (n_regions, n_regions)
    """

    # Transpose → (time, regions)
    data_matrix_t = data_matrix.T

    n_samples, n_regions = data_matrix_t.shape

    TE_mat = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(n_regions):

            if i == j:
                continue

            TE_mat[i, j] = transfer_entropy(
                data_matrix_t[:, i],   # source
                data_matrix_t[:, j],   # target
                method=method,
                **kwargs
            )

    return TE_mat


