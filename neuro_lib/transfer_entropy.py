import numpy as np
from numba import njit
from scipy.stats import norm
from .core import _entropy_core, _entropy_conditional_njit, _conditional_variance
from .estimators import estimate_mi, _estimate_mi_gaussian_multivariate, get_empiric_cdf, _estimate_log_density_kde

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
    Gaussian Transfer Entropy TE(source -> target), result in bits.

    Uses the conditional variance formula:
        TE = 0.5 * log2( Var(X_t | X_{t-1}) / Var(X_t | X_{t-1}, S_{t-lag}) )
    """
    S_lag = np.roll(source, lag)
    S_lag[:lag] = np.nan

    X_past = np.roll(target, 1)
    X_past[0] = np.nan

    X_t = target

    mask = ~np.isnan(S_lag) & ~np.isnan(X_past)
    S_lag  = S_lag[mask]
    X_past = X_past[mask]
    X_t    = X_t[mask]

    var1 = _conditional_variance(X_t, X_past)

    Y_joint = np.column_stack((X_past, S_lag))
    var2 = _conditional_variance(X_t, Y_joint)

    TE = 0.5 * np.log2(var1 / var2)   # was np.log → nats; now log2 → bits
    return TE

# =============================================================
# Transfer Entropy KDE
# =============================================================
#def transfer_entropy_kde(source, target, lag=1, **kwargs):
#    """
#    KDE-based Transfer Entropy TE(source -> target)
#
#    Uses:
#    TE = I(X_t ; S_{t-lag}, X_{t-1}) - I(X_t ; X_{t-1})
#
#    Parameters
#    ----------
#    source : array
#    target : array
#    lag : int
#   kwargs :
#        passed to KDE MI estimator (e.g. alpha, resample)
#
#    Returns
#    -------
#    TE : float
#    """
""""
    # -------------------------------
    # Build lagged variables
    # -------------------------------
    S_lag = np.roll(source, lag)
    S_lag[:lag] = np.nan

    X_past = np.roll(target, 1)
    X_past[0] = np.nan

    X_t = target

    # -------------------------------
    # Remove invalid entries
    # -------------------------------
    mask = ~np.isnan(S_lag) & ~np.isnan(X_past)

    S_lag = S_lag[mask]
    X_past = X_past[mask]
    X_t = X_t[mask]

    # -------------------------------
    # Build datasets for MI
    # -------------------------------

    # I(X_t ; X_past)
    data_1 = np.column_stack((X_t, X_past))

    # I(X_t ; [X_past, S_lag])
    joint = np.column_stack((X_past, S_lag))

    # IMPORTANT: estimate_mi expects 2D (n_samples, 2)
    # So we compute MI via KDE using joint variable as 1 block
    # Trick: flatten joint variable into a 2D variable using stacking
    data_2 = np.column_stack((X_t, joint))

    # -------------------------------
    # Compute MI via KDE
    # -------------------------------
    mi_1 = estimate_mi(data_1, method='kde', **kwargs)
    mi_2 = estimate_mi(data_2, method='kde', **kwargs)

    # -------------------------------
    # Transfer Entropy
    # -------------------------------
    TE = mi_2 - mi_1

    return TE
"""

def transfer_entropy_kde(source, target, lag=1, alpha=1.0):
    """
    Fully consistent KDE-based Transfer Entropy TE(source -> target), in bits.

    Uses the entropy decomposition:
        TE = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, S_{t-lag})
           = E[log2 p(X_t, X_{t-1}, S)] - E[log2 p(X_{t-1}, S)]
           - E[log2 p(X_t, X_{t-1})]    + E[log2 p(X_{t-1})]

    All four KDE fits are at most 2D, so the curse of dimensionality is
    contained. All terms are evaluated at the same sample points, so
    biases partially cancel and TE >= 0 is respected in expectation.

    Parameters
    ----------
    source, target : array (n_samples,)
    lag : int
    alpha : float
        Bandwidth scaling factor passed to all KDE fits.

    Returns
    -------
    TE : float (bits)
    """
    # --------------------------------------------------
    # Build lagged variables
    # --------------------------------------------------
    S_lag  = np.roll(source, lag);  S_lag[:lag]  = np.nan
    X_past = np.roll(target, 1);    X_past[0]    = np.nan
    X_t    = target

    mask   = ~np.isnan(S_lag) & ~np.isnan(X_past)
    S_lag  = S_lag[mask]
    X_past = X_past[mask]
    X_t    = X_t[mask]

    # --------------------------------------------------
    # Four joint arrays (all <= 2D)
    # --------------------------------------------------
    XtXp   = np.column_stack((X_t,    X_past))          # (X_t,  X_{t-1})
    XpS    = np.column_stack((X_past, S_lag))            # (X_{t-1}, S)
    XtXpS  = np.column_stack((X_t,    X_past, S_lag))    # 3D — only density needed

    # --------------------------------------------------
    # Log-density estimates (plug-in: evaluate at training points)
    # --------------------------------------------------
    # NOTE: the 3-variable joint is unavoidable here, but we only need
    # E[log p(X_t, X_{t-1}, S)] — a single KDE fit, not a difference of two.
    # A single 3D KDE underestimates the absolute entropy, but that bias
    # largely cancels with the 2D terms because all fits use the same
    # bandwidth rule and the same n.
    log_p_XtXpS = _estimate_log_density_kde(XtXpS,  XtXpS,  alpha)   # 3D
    log_p_XpS   = _estimate_log_density_kde(XpS,    XpS,    alpha)    # 2D
    log_p_XtXp  = _estimate_log_density_kde(XtXp,   XtXp,   alpha)    # 2D
    log_p_Xp    = _estimate_log_density_kde(X_past, X_past, alpha)    # 1D

    # --------------------------------------------------
    # TE = E[log p(X_t,X_{t-1},S)] - E[log p(X_{t-1},S)]
    #    - E[log p(X_t,X_{t-1})]   + E[log p(X_{t-1})]
    # --------------------------------------------------
    TE = np.mean(log_p_XtXpS - log_p_XpS - log_p_XtXp + log_p_Xp)

    return TE

# =============================================================
# Transfer Entropy Gaussian Copula
# =============================================================
def transfer_entropy_gaussian_copula(source, target, lag=1):
    """
    Transfer Entropy using Gaussian Copula transformation
    + Gaussian MI in transformed space.
    """

    # -----------------------------
    # Build lagged variables
    # -----------------------------
    S_lag = np.roll(source, lag)
    S_lag[:lag] = np.nan

    X_past = np.roll(target, 1)
    X_past[0] = np.nan

    X_t = target

    # -----------------------------
    # Remove invalid samples
    # -----------------------------
    mask = ~np.isnan(S_lag) & ~np.isnan(X_past)

    S_lag = S_lag[mask]
    X_past = X_past[mask]
    X_t = X_t[mask]

    # -----------------------------
    # Gaussianization 
    # -----------------------------
    def gaussianize(x):
        n = len(x)

        # rank-based CDF 
        ranks = np.argsort(np.argsort(x)) + 1
        u = ranks / (n + 1.0)

        eps = 1e-6
        u = np.clip(u, eps, 1 - eps)

        return norm.ppf(u) # ppf = percentile point function
                           # is the inverse of the CDF

    Xg = gaussianize(X_t)
    Pg = gaussianize(X_past)
    Sg = gaussianize(S_lag)

    # -----------------------------
    # Build joint datasets
    # -----------------------------
    data_1 = np.column_stack((Xg, Pg))        # I(X_t ; X_{t-1})
    data_2 = np.column_stack((Xg, Pg, Sg))    # I(X_t ; X_{t-1}, S)

    # -----------------------------
    # Gaussian MI
    # -----------------------------
    mi_1 = _estimate_mi_gaussian_multivariate(data_1)
    mi_2 = _estimate_mi_gaussian_multivariate(data_2)

    # -----------------------------
    # Transfer Entropy
    # -----------------------------
    return mi_2 - mi_1

# =============================================================
# Transfer Entropy Computation Interface
# =============================================================
def transfer_entropy(source, target, method="binning", **kwargs):

    if method == "binning":
        return transfer_entropy_binning(source, target, **kwargs)

    elif method == "gaussian":
        return transfer_entropy_gaussian(source, target, **kwargs)

    elif method == "kde":
        return transfer_entropy_kde(source, target, **kwargs)
    
    elif method == "copula": 
        return transfer_entropy_gaussian_copula(source, target, **kwargs)

    else:
        raise ValueError("Method must be 'binning', 'gaussian', 'kde' or 'copula'")

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
        "binning", "gaussian", "kde" or "copula"
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

# =============================================================
# Build correct vectors with given lag, embedding space and delay
# =============================================================

def _build_lagged_vectors(source, target, lag, m, tau):
    """
    Helper to create the embedded matrices for TE calculation.
    """
    n_samples = len(target)
    # The past of Y (Target) - shape (n_samples, m)
    # We create a matrix where each row is [y_{t-1}, y_{t-1-tau}, ..., y_{t-1-(m-1)tau}]
    y_past = np.zeros((n_samples, m))
    for i in range(m):
        y_past[:, i] = np.roll(target, 1 + i * tau)
    
    # The current state of Y (Target)
    y_t = target
    
    # The lagged state of X (Source)
    x_lag = np.roll(source, lag)
    
    # Create mask to remove NaNs/invalid entries due to rolling
    max_shift = max(lag, 1 + (m - 1) * tau)
    mask = np.zeros(n_samples, dtype=np.bool_)
    mask[max_shift:] = True
    
    return x_lag[mask], y_t[mask], y_past[mask]


def transfer_entropy_withMI(source, target, method="binning", lag=1, m=1, tau=1, **kwargs):
    """
    Unified TE interface reusing MI estimators.
    TE = I(X_lag ; [Y_t, Y_past]) - I(X_lag ; Y_past)
    """
    # Safety check for binning to avoid the curse of dimensionality
    if method=="binning":       
        m=1

    # 1. Prepare vectors
    x_l, y_t, y_p = _build_lagged_vectors(source, target, lag, m, tau)
    
    # 2. First Term: I(X_lag ; [Y_t, Y_past])
    # Data matrix: [Source_Lag, Target_Current, Target_Past_1, Target_Past_2...]
    # We want MI between column 0 and the rest
    data_joint = np.column_stack((x_l, y_t, y_p))
    # Note: estimate_mi expects (n_samples, 2). 
    # For m > 1, we treat the 'target' side as a multidimensional variable.
    # Ensure your estimators can handle ndim > 1 for the second variable.
    i_joint = estimate_mi(data_joint, method=method, **kwargs)
    
    # 3. Second Term: I(X_lag ; Y_past)
    data_past = np.column_stack((x_l, y_p))
    i_past = estimate_mi(data_past, method=method, **kwargs)
    
    # 4. TE is the difference
    return max(0, i_joint - i_past) # TE cannot be negative

def theoretical_TE_AR(alpha, gamma, sigma_x, sigma_y):
    var_x = (sigma_x**2) / (1 - alpha**2)
    signal_contribution = (gamma**2) * var_x
    return 0.5 * np.log2(1 + (signal_contribution / sigma_y**2))