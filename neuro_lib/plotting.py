import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



"""
This file should contain all the plotting functions
"""

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
    title_string="",
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

