import time
import numpy as np

from .line_search import backtracking_armijo


def gradient_descent_armijo(f, grad, x0,
                            alpha0=1.0, rho=0.5, c=1e-4,
                            maxiter=10000, tol=1e-6,
                            line_search=None, verbose=False):
    """
    Descenso por gradiente usando búsqueda de línea Armijo.
    Devuelve todos los datos necesarios para JSON y gráficas.
    """
    x = np.array(x0, dtype=float)
    f_evals = 0
    grad_evals = 0
    history = []
    start = time.time()

    for k in range(maxiter):
        g = grad(x)
        grad_evals += 1
        gnorm = np.linalg.norm(g)

        fx = float(f(x))
        f_evals += 1

        history.append({
            'k': k,
            'x': x.copy().tolist(),
            'f': fx,
            'grad_norm': gnorm
        })

        if verbose:
            print(f"Iter {k}: f = {fx:.6e}, ||grad|| = {gnorm:.2e}")

        if gnorm < tol:
            status = 'converged'
            break

        d = -g
        ls = backtracking_armijo(f, grad, x, d, alpha0=alpha0, rho=rho, c=c, verbose=verbose)
        alpha = ls['alpha']
        history[-1]['alpha'] = alpha

        x = x + alpha * d

    else:
        status = 'max_iter'

    end = time.time()
    x_final = x.copy()
    f_final = float(f(x_final))
    grad_norm_final = float(np.linalg.norm(grad(x_final)))
    grad_evals += 1

    result = {
        'x': x_final.tolist(),
        'f': f_final,
        'grad_norm': grad_norm_final,
        'history': history,
        'status': status,
        'time_seconds': end - start,
        'f_evals': f_evals,
        'grad_evals': grad_evals
    }
    return result
