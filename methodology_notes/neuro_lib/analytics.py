import numpy as np
from scipy.stats import gaussian_kde
from .generators import generate_data

"""
This is the place for all the functions which aim to compute (both analytically or numerically)
some information theoretical quantity (e.g. entropy, mi)
"""


# =============================================================
#                  Analytic InfoT Quantities
# =============================================================

def entropy_gaussian(std):
    return 0.5 * np.log2((2*np.pi*np.e * std**2))


def mi_gaussian_analytic(sigma_square, v_square = 1):
    """
    Analytic differential mutual information between two continuous
    normal variables with variances $\sigma^2$ and $v^2$.
    """
    return 0.5 * np.log2(1 + v_square/sigma_square)


def entropy_pmf(p):
    #Computes Shannon entropy of a probability mass function (p.m.f.)
    #p: probability vector (entries sum up to 1)
    if abs(np.sum(p) - 1.0) > 1e-6:
        raise ValueError("PMF not normalized")
    p = p[p > 1e-6]
    return  - np.sum(np.log2(p) * p)



# =============================================================
#                      Numeric InfoT Quantities
# =============================================================


def entropy_numeric(f, N_large= int(1e3), sigma=0.1, return_model = False):
    """
    Numeric calculation of differential shannon entropy for X
    X = f(S) + sigma * eta
    S: normal random variable of std = 1, mean = 0
    f: arbitrary function
    eta: standard gaussian noise (mean 0, std 1)
    sigma: gaussian channel noise amplitude
    """
    data = generate_data(N_large, sigma, f)
    kde = gaussian_kde(data[: ,1], bw_method='scott')

    # H(X) = -E[log(p(x))]
    X_new = kde.resample(size = N_large, seed = 0)
    log2_pdf = kde.logpdf(X_new) / np.log(2)
    entropy = -np.mean(log2_pdf)
    if return_model:
        return kde, entropy
    return entropy

def mi_numeric(f, N_large= 1000, sigma=0.1, return_model = False):
    hx = entropy_numeric(f, N_large, sigma, return_model)
    return hx - entropy_gaussian(sigma)

