"""
HELPERS MODULE FOR ITIPRroject.ipynb

CONTAINS:
--------

.... to complete
    --> generate_neural_variable: Generate a synthetic neural variable X(t) given an external signal S(t)
        with optional lag, auto-regression, and nonlinear relation.

    --> plot_ts_and_hist: Plots a time series and histogram of the input data.

    --> plot_joint_distribution: Plot the joint distribution of S (x-axis) and X (y-axis).

"""

import numpy
import matplotlib.pyplot
import seaborn
from scipy.signal import savgol_filter
from numba import njit

# =============================================================
#                      Data Generators
# =============================================================

def generate_neural_variable(
    S, 
    a=1.0, 
    noise_std=1.0, 
    lag=False, 
    tau=None, 
    auto_regr=False, 
    phi=None, 
    relation="linear"
):
    """
    Generate a synthetic neural variable X(t) given an external signal S(t)
    with optional lag, auto-regression, and nonlinear relation.

    Parameters
    ----------
    S : array_like
        External signal of length Ntsteps.
    a : float
        Coefficient linking S to X.
    noise_std : float
        Standard deviation of Gaussian noise.
    lag : bool
        If True, X depends on a lagged version of S (S[t-tau]).
    tau : int
        Lag in time steps (used if lag=True or auto_regr=True).
    auto_regr : bool
        If True, generate an auto-regressive process X[t] = phi*X[t-1] + a*f(S[t-tau]) + noise.
    phi : float
        AR(1) coefficient (used if auto_regr=True).
    relation : str
        Type of relation between S and X. Options: "linear", "cubic", "quadratic", "tanh"

        
    General expression:

        X[t] = phi * X[t-1] + a * f(S[t - tau]) + eta[t] (gaussian white noise)

    Returns
    -------
    X : numpy.ndarray
        Generated neural variable.
    """

    Ntsteps = len(S)
    eta = numpy.random.normal(0, noise_std, Ntsteps)

    # Define the function f(S) based on relation
    def f(S_val):
        if relation == "linear":
            return S_val
        elif relation == "cubic":
            return S_val**3
        elif relation == "quadratic":
            return 2*S_val**2 - 1
        elif relation == "tanh":
            return numpy.tanh(5 * S_val)
        else:
            raise ValueError(f"Unknown relation: {relation}")

    # simple linear or lagged linear
    if not auto_regr:
        if lag:
            S_lag = numpy.roll(S, tau) 
            # NOTICE: numpy.roll(S, tau) shifts the array S forward tau steps
            #         E.g. if S = [1,2,3,4] and tau = 1 
            #              then numpy.roll(S,1) --> [4,1,2,3]
            #         But for lag we don't want to have the first tau elements
            S_lag[:tau] = numpy.nan  # avoid wrap-around
            X = a * f(S_lag) + eta
        else:
            X = a * f(S) + eta
    else:
        # AR with optional lagged input
        X = numpy.zeros(Ntsteps)
        for t in range(1, Ntsteps):
            S_val = S[t-tau] if lag else S[t]
            X[t] = phi * X[t-1] + a * f(S_val) + eta[t]

    return X



# =============================================================
#                  Analytic InfoT Quantities
# =============================================================

def mi_gaussian_analytic(sigma_square, v_square = 1):
    """
    Analytic differential mutual information between two continuous
    normal variables with variances $\sigma^2$ and $v^2$.
    """
    return 0.5 * numpy.log2(1 + v_square/sigma_square)


def entropy_pmf(p):
    #Computes Shannon entropy of a probability mass function (p.m.f.)
    #p: probability vector (entries sum up to 1)

    if abs(numpy.sum(p) - 1.0) > 1e-6:
        raise ValueError("PMF not normalized")
    p = p[p > 1e-6]
    return  - numpy.sum(numpy.log2(p) * p)

@njit(cache=True)
def entropy_pmf_numba(p):

    #Computes Shannon entropy of a probability mass function (p.m.f.)
    #p: probability vector (entries sum up to 1)
    if abs(numpy.sum(p) - 1.0) > 1e-6:
        raise ValueError("PMF not normalized")
    entropy = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            entropy -= p[i] * numpy.log2(p[i])
    return entropy

# =============================================================
#                      Estimators
# =============================================================

# Discrete Shannon Entropy estimator

@njit(cache=True)
def _entropy_core(counts, bias_correction = None):
    """
    Estimates entropy of a probability mass function (pmf)
    from empirical frequencies
    counts: empirical observations, divided per category (bins, classes, whathever)
    """
    total = numpy.sum(counts)
    if total == 0:
        return 0.0
    res = 0.0
    m = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            m += 1
            p = counts[i] / total
            res -= p * numpy.log2(p)
    if bias_correction == "miller":
        res += (m - 1)/(2 * total)
    return res


# Binning Method
"""
def entropy_binning_1d(data, bin_number, which = "absolute", bias_correction = None):
    #Discretizes continuous data and computes the shannon entropy of the corresponding p.m.f.
    #data: 2d numpy array, nrows = number of observations
    #which: 
    #    - "absolute" computes h(x)
    #    - "conditional": computes h(x|s)
    #
    S = data[:, 0]
    X = data[:, 1]
    N_observations = data.shape[0]

    if which == "absolute":
        counts, _ = numpy.histogram(X, bin_number, density = False)
        return entropy_pmf(frequencies)
    
    if which == "conditional":
        # Pre-calculate global bins for BOTH X and S
        _, s_edges = numpy.histogram(S, bins=bin_number)
        _, x_edges = numpy.histogram(X, bins=bin_number)
        
        H_x_given_s = 0.0
        
        for i in range(bin_number):
            mask = (S >= s_edges[i]) & (S < s_edges[i+1])
            if i == bin_number - 1:
                mask = (S >= s_edges[i]) & (S <= s_edges[i+1])
                
            subset = X[mask]
            if subset.size == 0:
                continue
            
            counts, _ = numpy.histogram(subset, bins=x_edges)
            p_x_given_si = counts / subset.size
            p_si = subset.size / N_observations
            H_x_given_s += p_si * entropy_pmf(p_x_given_si)
        return H_x_given_s
"""

# --------------------------

@njit(cache=True)
def _bin_data_1d(x, n_bins, x_min, x_max):
    """Manual 1D histogramming."""
    counts = numpy.zeros(n_bins, dtype=numpy.int64)
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
def _entropy_conditional_njit(S, X, n_bins, bias_correction = None):
    n_obs = len(S)
    s_min, s_max = S.min(), S.max()
    x_min, x_max = X.min(), X.max()
    
    s_width = (s_max - s_min) / n_bins if s_max > s_min else 1.0
    x_width = (x_max - x_min) / n_bins if x_max > x_min else 1.0
    

    counts_2d = numpy.zeros((n_bins, n_bins), dtype=numpy.int64)
    s_bin_totals = numpy.zeros(n_bins, dtype=numpy.int64)
    
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

def entropy_binning_1d_numba(data, bin_number, which="absolute", bias_correction = None):
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
# -----------------------

"""
def mi_binning_2d(data, bin_number):
    #Discretizes continuous data and computes the mutual information of the corresponding pmf
    S = data[:, 0]
    X = data[:, 1]
    Hx = entropy_binning_1d(data, bin_number)
    Hx_given_s = entropy_binning_1d(data, bin_number, "conditional")
    return Hx - Hx_given_s
"""

def mi_binning_2d_numba(data, bin_number, bias_correction = None):
    """
    Discretizes continuous data and computes the mutual information of the corresponding pmf
    """
    S = data[:, 0]
    X = data[:, 1]
    Hx = entropy_binning_1d_numba(data, bin_number, "absolute" ,bias_correction)
    Hx_given_s = entropy_binning_1d_numba(data, bin_number, "conditional", bias_correction)
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
        return 1.0 / numpy.sqrt(n_obs)
        
    counts = numpy.zeros(bin_number, dtype=numpy.int64)
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
            total_error += 1.0 / numpy.sqrt(c)
            active_bins += 1
    if active_bins == 0:
        return 0.0   
    return total_error / active_bins


# Joint Normality Assumption

def mi_gaussian_estimator(X, S):
    """
    Estimator from data
    Compute mutual information (in bits) assuming joint Gaussian distribution.

    Parameters
    ----------
    X, S : array_like
        Two 1D arrays of the same length

    Returns
    -------
    MI : float
        Mutual information between X and S
    """
    rho = numpy.corrcoef(X, S)[0,1]  # Pearson correlation
    MI = -0.5 * numpy.log2(1 - rho**2)
    return MI

# TODO Gaussian Copula Assumption

# TODO KDE Estimation

# =============================================================
#                      Plotting Fucntions
# =============================================================

def plot_ts(data, title_string, color = "steelblue"):
    """
    Plots a time series and histogram of the input data.
    
    Parameters
    ----------
    data : array-like
        The time series or signal to plot.
    title_string : str
        The object of which we're plotting the ts and histogram. 
    color: str, optional
        Chosen color
    """
    # Plot time series
    matplotlib.pyplot.figure(figsize=(6, 3))
    matplotlib.pyplot.plot(data, lw=0.5, color=color)
    matplotlib.pyplot.title(f"Time series of {title_string}")
    matplotlib.pyplot.xlabel("t")
    matplotlib.pyplot.ylabel("S(t)")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def plot_hist(data, title_string, color = "steelblue", xlabel = None):    
    # Plot histogram
    matplotlib.pyplot.figure(figsize=(5, 3))
    matplotlib.pyplot.hist(data, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    matplotlib.pyplot.title(f"Histogram of {title_string}")
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel("Probability density")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

# =============================================================
def plot_joint_distribution(
    S,
    X,
    title_string="Joint distribution",
    color="steelblue",
    bins=50,
    density=False
):
    """
    Plot the joint distribution of S (x-axis) and X (y-axis).

    Parameters
    ----------
    S : array-like
        External signal.
    X : array-like
        Neural variable.
    title_string : str
        Title of the plot.
    color : str, optional
        Scatter color.
    bins : int, optional
        Number of bins for 2D histogram (used if density=True).
    density : bool, optional
        If True, plots a 2D histogram instead of scatter.
    """

    # Remove NaNs if present (important for lag case)
    # NOTICE: does not break pairing because we're only keeping data where 
    #         both X and S are valid
    mask = ~numpy.isnan(S) & ~numpy.isnan(X) #~ is bitwise NOT operatore for Boolean arrays
    S_clean = S[mask]
    X_clean = X[mask]

    matplotlib.pyplot.figure(figsize=(4, 4))

    if density:
        matplotlib.pyplot.hist2d(S_clean, X_clean, bins=bins, density=True)
        matplotlib.pyplot.colorbar(label="Density")
    else:
        matplotlib.pyplot.scatter(S_clean, X_clean, s=8, alpha=0.4, color=color)

    matplotlib.pyplot.xlabel("S(t)")
    matplotlib.pyplot.ylabel("X(t)")
    matplotlib.pyplot.title(title_string)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()


# =============================================================
def plot_joint_distribution_sns(
    S,
    X,
    title_string="",
    color="steelblue",
    kind="scatter", # Options: 'scatter', 'hist', 'kde', 'hex'
    bins=50
):
    """
    Plot the joint distribution of S and X with marginals using Seaborn.
    """
    # Clean NaNs to prevent Seaborn from plotting empty frames
    mask = ~numpy.isnan(S) & ~numpy.isnan(X)
    S_clean = S[mask]
    X_clean = X[mask]

    # Create the joint plot
    # 'kind' replaces your density toggle: 'scatter' for points, 'hist' for 2D histogram
    g = seaborn.jointplot(
        x=S_clean, 
        y=X_clean, 
        kind=kind, 
        color=color,
        marginal_kws=dict(bins=bins, fill=True)
    )

    # Set labels and title
    g.set_axis_labels("S(t)", "X(t)")
    g.fig.suptitle("Joint Distribution - " + title_string, y=1.02) # Adjust title to not overlap marginals

    g.ax_joint.grid(True, linestyle='--', alpha=0.6)
    matplotlib.pyplot.show()
