"""
HELPERS MODULE FOR ITIPRroject.ipynb

CONTAINS:
--------
    --> generate_neural_variable: Generate a synthetic neural variable X(t) given an external signal S(t)
        with optional lag, auto-regression, and nonlinear relation.

    --> plot_ts_and_hist: Plots a time series and histogram of the input data.

    --> plot_joint_distribution: Plot the joint distribution of S (x-axis) and X (y-axis).

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    X : np.ndarray
        Generated neural variable.
    """

    Ntsteps = len(S)
    eta = np.random.normal(0, noise_std, Ntsteps)

    # Define the function f(S) based on relation
    def f(S_val):
        if relation == "linear":
            return S_val
        elif relation == "cubic":
            return S_val**3
        elif relation == "quadratic":
            return S_val**2
        elif relation == "tanh":
            return np.tanh(S_val)
        else:
            raise ValueError(f"Unknown relation: {relation}")

    # simple linear or lagged linear
    if not auto_regr:
        if lag:
            S_lag = np.roll(S, tau) 
            # NOTICE: np.roll(S, tau) shifts the array S forward tau steps
            #         E.g. if S = [1,2,3,4] and tau = 1 
            #              then np.roll(S,1) --> [4,1,2,3]
            #         But for lag we don't want to have the first tau elements
            S_lag[:tau] = np.nan  # avoid wrap-around
            X = a * f(S_lag) + eta
        else:
            X = a * f(S) + eta
    else:
        # AR with optional lagged input
        X = np.zeros(Ntsteps)
        for t in range(1, Ntsteps):
            S_val = S[t-tau] if lag else S[t]
            X[t] = phi * X[t-1] + a * f(S_val) + eta[t]

    return X

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
    plt.figure(figsize=(6, 3))
    plt.plot(data, lw=0.5, color=color)
    plt.title(f"Time series of {title_string}")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.tight_layout()
    plt.show()

def plot_hist(data, title_string, color = "steelblue", xlabel = None):    
    # Plot histogram
    plt.figure(figsize=(5, 3))
    plt.hist(data, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    plt.title(f"Histogram of {title_string}")
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.tight_layout()
    plt.show()

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
    mask = ~np.isnan(S) & ~np.isnan(X) #~ is bitwise NOT operatore for Boolean arrays
    S_clean = S[mask]
    X_clean = X[mask]

    plt.figure(figsize=(4, 4))

    if density:
        plt.hist2d(S_clean, X_clean, bins=bins, density=True)
        plt.colorbar(label="Density")
    else:
        plt.scatter(S_clean, X_clean, s=8, alpha=0.4, color=color)

    plt.xlabel("S(t)")
    plt.ylabel("X(t)")
    plt.title(title_string)
    plt.tight_layout()
    plt.show()


# =============================================================
def plot_joint_distribution_sns(
    S,
    X,
    title_string=None,
    color="steelblue",
    kind="scatter", # Options: 'scatter', 'hist', 'kde', 'hex'
    bins=50
):
    """
    Plot the joint distribution of S and X with marginals using Seaborn.
    """
    # Clean NaNs to prevent Seaborn from plotting empty frames
    mask = ~np.isnan(S) & ~np.isnan(X)
    S_clean = S[mask]
    X_clean = X[mask]

    # Create the joint plot
    # 'kind' replaces your density toggle: 'scatter' for points, 'hist' for 2D histogram
    g = sns.jointplot(
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
    plt.show()

# =============================================================
def gaussian_mi(X, S):
    """
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
    rho = np.corrcoef(X, S)[0,1]  # Pearson correlation
    MI = -0.5 * np.log2(1 - rho**2)
    return MI