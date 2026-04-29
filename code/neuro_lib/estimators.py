import numpy as np
from scipy.stats import norm, gaussian_kde
from kneed import KneeLocator

from .core import mi_binning_2d_numba


"""
This is the place for all functions that ESTIMATE some information 
theoretical quantity from finite size DATA
"""

# =============================================================
#                      Internal Estimators
# =============================================================


# ----------------------- Binning -----------------------------
def _estimate_mi_binning(data, bins):
    y = np.array([mi_binning_2d_numba(data, b) for b in bins])
    kn = KneeLocator(bins, y, S=1.0, curve='concave', direction='increasing')
        
    if kn.knee is not None:
        knee_idx = np.argwhere(bins == kn.knee)[0][0]
        res = y[knee_idx]
    else:
        N = data.shape[0]
        fallback_bin = int(np.sqrt(N))     # heuristic: sqrt rule 
        idx = np.argmin(np.abs(bins - fallback_bin))
        res=y[idx]

    return res

# ----------------------- Gaussian Copula ----------------------

def get_empiric_cdf(data):
    """
    Computes empirical cdf from 'data'
    returns a callable function that returns the empirical cdf value for an arbitrary input value x
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    def cdf(x):
        count = np.searchsorted(sorted_data, x, side='right') # count data <= x
        return count / n
    return cdf

def _estimate_mi_gaussian_copula(data):
    # Apply empirical CDF transform to each column
    transformed_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        cdf = get_empiric_cdf(data[:, i])
        u = cdf(data[:, i])
        transformed_data[:, i] = norm.ppf(np.clip(u, 1e-15, 1 - 1e-15))
    
    # Now call the updated Gaussian MI
    return _estimate_mi_gaussian(transformed_data)

def _estimate_mi_gaussian_copula1D(data):
    "data: nrows = number of samples, ncols = 2"

    x = data[:, 0]
    y = data[:, 1]

    # 1. compute empirical cdfs
    cdfx = get_empiric_cdf(x)
    cdfy = get_empiric_cdf(y)
    u = cdfx(x)
    v = cdfy(y)

    # 2. transform X-> X' and Y->Y'
    eps = 1e-15 # cap to prevent infinities in xprime, yprime
    xprime = norm.ppf(np.clip(u, eps, 1 - eps)) # ppf= percent point function (inverse cdf)
    yprime = norm.ppf(np.clip(v, eps, 1 - eps))
    dataprime = np.vstack([xprime, yprime]) # nrows = 2, ncols = nobs.

    # 3. compute gaussian mi
    rho = np.corrcoef(dataprime, rowvar = True)[0][1]
    mi_GC =  -0.5 * np.log2(1 - rho**2)

    return mi_GC
# ----------------------- KDE  ----------------------

def estimate_entropy_kde(variable_vec):
    kde = gaussian_kde(variable_vec)
    return - np.mean(kde.logpdf(variable_vec)/np.log(2))

def _estimate_mi_kde(data, alpha=1.0):
    # data[:, 0] is X, data[:, 1:] is Y
    x = data[:, 0]
    y = data[:, 1:]
    
    # KDE for X, Y, and Joint
    kde_x = gaussian_kde(x)
    kde_y = gaussian_kde(y.T)
    kde_joint = gaussian_kde(data.T)
    
    # Apply bandwidth tuning if needed
    for k in [kde_x, kde_y, kde_joint]:
        k.set_bandwidth(bw_method=k.factor * alpha)

    log_f_x = kde_x.logpdf(x) / np.log(2)
    log_f_y = kde_y.logpdf(y.T) / np.log(2)
    log_f_joint = kde_joint.logpdf(data.T) / np.log(2)
    
    return np.mean(log_f_joint - log_f_x - log_f_y)

def _estimate_mi_kde1D(data, resample = False, alpha=1.0):
    s = data[:, 0]
    x = data[:, 1]

     # --- KDEs ---
    kde_s = gaussian_kde(s)
    kde_x = gaussian_kde(x)
    kde_sx = gaussian_kde(data.T)

    # --- Bandwidth tuning ---
    kde_s.set_bandwidth(bw_method=kde_s.factor * alpha)
    kde_x.set_bandwidth(bw_method=kde_x.factor * alpha)
    kde_sx.set_bandwidth(bw_method=kde_sx.factor * alpha)

    # --- log densities ---
    log_f_s = kde_s.logpdf(s) / np.log(2)
    log_f_x = kde_x.logpdf(x) / np.log(2)
    log_f_sx = kde_sx.logpdf(data.T) / np.log(2)
    mi = np.mean(log_f_sx - log_f_s - log_f_x)
    return mi

# ----------------------- Gaussian ----------------------

def _estimate_mi_gaussian(data):
    """
    Handles X (col 0) and Y (cols 1:). 
    Works for any dimension of Y.
    """
    C = np.cov(data, rowvar=False)
    
    # Variance of X (scalar)
    det_X = C[0, 0]
    # Determinant of Covariance of Y
    det_Y = np.linalg.det(C[1:, 1:])
    # Determinant of Joint Covariance
    det_Joint = np.linalg.det(C)
    
    # Avoid log of zero or negative due to noise
    val = (det_X * det_Y) / det_Joint
    return 0.5 * np.log2(max(val, 1e-10))

def _estimate_mi_gaussian1D(data):
    """
    Estimator from data
    Compute mutual information (in bits) assuming joint Gaussian distribution.

    data: nrows = number of samples, ncols = 2
    """
    rho = np.corrcoef(data, rowvar = False)[0,1]
    return -0.5 * np.log2(1 - rho**2)


# =============================================================
#                      Public Wrapper (API)
# =============================================================

MI_ESTIMATION_METHODS = {
    'binning': _estimate_mi_binning,
    'kde': _estimate_mi_kde,
    'gc': _estimate_mi_gaussian_copula,
    'gauss': _estimate_mi_gaussian,
}

def estimate_mi(data, method='binning', **kwargs):
    """
    Wrapper for MI estimation from data
    """
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty.")
    if method not in MI_ESTIMATION_METHODS:
        raise ValueError(f"Method {method} not recognized.")
    return MI_ESTIMATION_METHODS[method](data, **kwargs)



# EXTENSION OF GAUSSIAN MI FOR COPULA
def _estimate_mi_gaussian_multivariate(data, eps=1e-10):
    """
    Gaussian Mutual Information for multivariate case:
    I(X;Y) = 1/2 log2( det Σ_X det Σ_Y / det Σ_{XY} )
    """

    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1:]

    # Covariances
    # np.atleast_2d ensures the argument is at least 2D
    # e.g. x = np.array([1,2,3]) i.e. shape = (3,)
    # np.atleast_2d(x) --> array([[1,2,3]]) i.e. shape = (1,3)
    cov_X = np.atleast_2d(np.cov(X, rowvar=False))
    cov_Y = np.atleast_2d(np.cov(Y, rowvar=False))
    cov_XY = np.atleast_2d(np.cov(data, rowvar=False))

    # Regularization 
    cov_X = cov_X + eps * np.eye(cov_X.shape[0])
    cov_Y = cov_Y + eps * np.eye(cov_Y.shape[0])
    cov_XY = cov_XY + eps * np.eye(cov_XY.shape[0])

    # Determinants
    det_X = np.linalg.det(cov_X)
    det_Y = np.linalg.det(cov_Y)
    det_XY = np.linalg.det(cov_XY)

    # Numerical safety
    det_X = max(det_X, eps)
    det_Y = max(det_Y, eps)
    det_XY = max(det_XY, eps)

    mi = 0.5 * np.log2((det_X * det_Y) / det_XY)

    return mi


# EXTENSIONS FOR KDE
def _estimate_log_density_kde(eval_points, fit_points, alpha=1.0):
    """
    Fit a KDE on fit_points, evaluate log-density at eval_points.
    Returns array of log2-densities at each eval point.

    Parameters
    ----------
    eval_points : array (n_samples,) or (n_samples, d)
    fit_points  : array (n_samples,) or (n_samples, d)
        Must be same shape. In practice we always evaluate at the
        training points (plug-in estimator).
    alpha : float
        Bandwidth scaling factor.

    Returns
    -------
    log2_density : array (n_samples,)
    """
    # gaussian_kde expects shape (d, n)
    if fit_points.ndim == 1:
        fit_T = fit_points
        eval_T = eval_points
    else:
        fit_T  = fit_points.T
        eval_T = eval_points.T

    kde = gaussian_kde(fit_T)
    kde.set_bandwidth(bw_method=kde.factor * alpha)
    return kde.logpdf(eval_T) / np.log(2)   # nats → bits
