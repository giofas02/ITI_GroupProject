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


# =============================================================
def TE_heatMat(TE_mat, figsize=(10,8), cmap='viridis', show=True, n_ticks=5):
    """
    Plot a heatmap of a Transfer Entropy (TE) matrix with limited axis ticks.

    Parameters
    ----------
    TE_mat : np.ndarray
        Square matrix of transfer entropy (n_regions x n_regions).
    figsize : tuple, default (10, 8)
        Size of the figure.
    cmap : str, default 'viridis'
        Colormap for the heatmap.
    show : bool, default True
        Whether to call plt.show() immediately.
    n_ticks : int, default 5
        Number of ticks to display on each axis (evenly spaced).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object of the heatmap.
    """
    n_regions = TE_mat.shape[0]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        TE_mat,
        cmap=cmap,
        center = 0,
        square=True,
        vmin = 0, 
        vmax = 0.25, 
        cbar_kws={'label': 'Transfer Entropy'},
        xticklabels=False,  # will set ticks manually
        yticklabels=False
    )

    # Set limited ticks
    tick_positions = np.linspace(0, n_regions-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions + 0.5)  # +0.5 to center tick on the cell
    ax.set_yticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)

    ax.set_title('Transfer Entropy Matrix Heatmap')
    ax.set_xlabel('Target Region')
    ax.set_ylabel('Source Region')

    if show:
        plt.show()
    return ax


# =============================================================
#               Plotting Diagnostics of Time series
# =============================================================

def plot_diagnostic_ami(lags, mi_values, tau_opt):
    plt.figure(figsize=(6, 4))
    plt.plot(lags, mi_values, marker='o', markersize=4, label='AMI')
    plt.axvline(tau_opt, color='red', linestyle='--', label=f'Tau Opt: {tau_opt}')
    plt.title("Average Mutual Information for Delay Selection")
    plt.xlabel("Lag")
    plt.ylabel("MI (bits)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_diagnostic_fnn(m_range, fnn_percentages, m_opt):
    plt.figure(figsize=(6, 4))
    plt.plot(m_range, fnn_percentages, marker='s', color='orange', label='FNN %')
    plt.axvline(m_opt, color='red', linestyle='--', label=f'm Opt: {m_opt}')
    plt.axhline(1.0, color='black', linestyle=':', label='1% Threshold')
    plt.title("False Nearest Neighbors for Dimension Selection")
    plt.xlabel("Dimension (m)")
    plt.ylabel("FNN %")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()