import time
import numpy as np
from scipy.optimize import line_search as _line_search

def bfgs(f, grad, x0, H0=None, tol=1e-6, maxiter=200,
         line_search='wolfe', use_damping=True, skip_update=False,
         verbose=False):
    """
    Método BFGS con registro completo para serialización y graficación.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    if H0 is None:
        H = np.eye(n)
    else:
        H = np.asarray(H0, dtype=float).copy()
    H = 0.5 * (H + H.T)

    f_evals = 0
    grad_evals = 0
    history = []
    start = time.time()

    for k in range(maxiter):
        g = grad(x)
        grad_evals += 1
        gnorm = np.linalg.norm(g)

        fval = float(f(x))
        f_evals += 1

        alpha_used = None
        ls_iters_used = None

        history.append({
            'k': k,
            'x': x.copy().tolist(),
            'f': fval,
            'grad_norm': gnorm,
        })

        if verbose:
            print(f"Iter {k}: f = {fval:.6e}, ||∇f|| = {gnorm:.2e}")

        if gnorm <= tol:
            status = 'converged'
            break

        d = - H.dot(g)
        if np.dot(d, g) >= 0:
            d = -g
            if verbose:
                print(f"  [Iter {k}] Dirección no es de descenso → uso -gradiente")

        if line_search == 'wolfe':
            try:
                alpha_w, _, _ = _line_search(f, grad, x, d, gfk=g,
                                             old_fval=fval, c1=1e-4, c2=0.9)
                if alpha_w is None or not np.isfinite(alpha_w) or alpha_w <= 0:
                    raise ValueError
                alpha_used = float(alpha_w)
                ls_iters_used = 0
            except Exception:
                from optimizacion.line_search import backtracking_armijo
                arm = backtracking_armijo(f, grad, x, d, verbose=verbose)
                alpha_used = arm['alpha']
                ls_iters_used = arm['ls_iters']
        else:
            from optimizacion.line_search import backtracking_armijo
            arm = backtracking_armijo(f, grad, x, d, verbose=verbose)
            alpha_used = arm['alpha']
            ls_iters_used = arm['ls_iters']

        history[-1]['alpha'] = alpha_used
        history[-1]['ls_iters'] = ls_iters_used

        if verbose:
            print(f"  → paso α = {alpha_used:.3e}, ls_iters = {ls_iters_used}")

        s = alpha_used * d
        x_new = x + s

        g_new = grad(x_new)
        grad_evals += 1

        y = g_new - g
        sty = float(np.dot(s, y))

        if sty <= 0 or not np.isfinite(sty):
            if skip_update and not use_damping:
                if verbose:
                    print(f"  [Skip update: sᵀy = {sty:.2e}]")
                H_new = H
            elif use_damping:
                Hs = H.dot(s)
                sTHs = float(np.dot(s, Hs))
                if sTHs > 0:
                    if sty >= 0.2 * sTHs:
                        theta = 1.0
                    else:
                        theta = (0.8 * sTHs) / (sTHs - sty)
                        if verbose:
                            print(f"  [Damping Powell θ={theta:.3f}]")
                    y_tilde = theta * y + (1.0 - theta) * Hs
                    sty_tilde = float(np.dot(s, y_tilde))
                    if sty_tilde > 0:
                        rho2 = 1.0 / sty_tilde
                        I = np.eye(n)
                        V = I - rho2 * np.outer(s, y_tilde)
                        H_new = V.dot(H).dot(V.T) + rho2 * np.outer(s, s)
                    else:
                        H_new = H
                else:
                    H_new = H
            else:
                H_new = H
        else:
            rho2 = 1.0 / sty
            I = np.eye(n)
            V = I - rho2 * np.outer(s, y)
            H_new = V.dot(H).dot(V.T) + rho2 * np.outer(s, s)

        x = x_new
        H = 0.5 * (H_new + H_new.T)

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
