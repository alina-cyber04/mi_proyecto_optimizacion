"""Paquete optimizacion.

Exporta las funciones principales de los módulos del subpaquete.
"""

from .line_search import backtracking_armijo
from .descent import gradient_descent_armijo
from .bfgs import bfgs
from .util_json import serialize_experiment_to_json, load_experiments_from_json
from .graficos import plot_convergence, plot_final_vs_iters, plot_trajectory_2d

__all__ = [
    "backtracking_armijo",
    "gradient_descent_armijo",
    "bfgs",
    "serialize_experiment_to_json",
    "load_experiments_from_json",
    "plot_convergence",
    "plot_final_vs_iters",
    "plot_trajectory_2d",
]
