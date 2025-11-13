"""Búsqueda de línea por retroceso (Armijo).
"""
import time
import numpy as np


def backtracking_armijo(f, grad, x, d,
                        alpha0=1.0, rho=0.5, c=1e-4,
                        max_iter=50, alpha_min=1e-8, verbose=False):
    """
    Búsqueda de línea por retroceso (Armijo).
    Retorna un dict {'alpha': valor, 'ls_iters': número de iteraciones de línea de búsqueda}.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    alpha = float(alpha0)

    fx = float(f(x))
    gx = np.asarray(grad(x), dtype=float)
    phi_prime0 = float(np.dot(gx, d))
    if phi_prime0 >= 0:
        raise ValueError(f"La dirección no es de descenso: ∇f(x)ᵀ d = {phi_prime0:.3e}")

    ls_iters = 0
    while ls_iters < max_iter and alpha > alpha_min:
        x_new = x + alpha * d
        try:
            f_new = float(f(x_new))
        except Exception:
            f_new = np.inf

        if verbose:
            print(f"[Armijo] ls_iter={ls_iters}, α={alpha:.3e}, f_new={f_new:.3e}, rhs={fx + c*alpha*phi_prime0:.3e}")

        if f_new <= fx + c * alpha * phi_prime0:
            break

        alpha *= rho
        ls_iters += 1

    return {'alpha': float(alpha), 'ls_iters': ls_iters}
