# neuro_lib/__init__.py

# Hoist the most important "Public API" functions
from .estimators import estimate_mi, get_empiric_cdf, estimate_entropy_kde
from .generators import generate_neural_variable, generate_data
from .analytics import entropy_numeric, mi_numeric, entropy_gaussian, mi_gaussian_analytic
from .plotting import (
    plot_ts, 
    plot_hist, 
    plot_joint_distribution, 
    plot_joint_distribution_sns
)

# Optional: Define what is exported when someone does 'from mi_engine import *'
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
    'plot_joint_distribution_sns'
]