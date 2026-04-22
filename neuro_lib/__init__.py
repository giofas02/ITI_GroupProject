# neuro_lib/__init__.py

# Hoist the most important "Public API" functions
from .estimators import estimate_mi, get_empiric_cdf, estimate_entropy_kde
from .generators import generate_neural_variable, generate_data, generate_ar_coupled, generate_oscillatory_coupled
from .analytics import entropy_numeric, mi_numeric, entropy_gaussian, mi_gaussian_analytic
from .plotting import (plot_ts, plot_hist, plot_joint_distribution, plot_joint_distribution_sns, plot_diagnostic_ami, plot_diagnostic_fnn)
from .transfer_entropy import (
    transfer_entropy_matrix,
    transfer_entropy_binning, 
    transfer_entropy_gaussian,
    transfer_entropy_kde, 
    transfer_entropy_gaussian_copula, transfer_entropy_withMI, theoretical_TE_AR
)
from .diagnostics import check_stationarity, find_optimal_delay, find_optimal_dimension, test_significance, permutation_test_TE

# Optional: Define what is exported when someone does 'from neuro_lib import *'
__all__ = [
    'get_empiric_cdf',
    'estimate_mi',
    'estimate_entropy_kde',
    'generate_neural_variable',
    'generate_data',
    'mi_numeric',
    'entropy_numeric',
    'entropy_gaussian',
    'mi_gaussian_analytic',
    'plot_joint_distribution_sns',
    'transfer_entropy_matrix',
    'transfer_entropy_binning',
    'transfer_entropy_gaussian',
    'transfer_entropy_kde', 
    'transfer_entropy_gaussian_copula',
        'generate_ar_coupled',
    'generate_oscillatory_coupled',
    'check_stationarity',
    'find_optimal_delay',
    'find_optimal_dimension', 
    'plot_diagnostic_ami', 
    'plot_diagnostic_fnn',
    'transfer_entropy_withMI',
    'theoretical_TE_AR',
    'test_significance', 
    'permutation_test_TE'
]