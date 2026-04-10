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

def _estimate_mi_kde(data, resample = False, alpha=1.0):
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