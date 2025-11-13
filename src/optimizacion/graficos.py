import matplotlib.pyplot as plt
import numpy as np
import hashlib
from collections import OrderedDict


def _plot_convergence_single(experiments, ax_f, ax_grad, log_scale=True,
                             legend_fontsize='small', compact_labels=True,
                             smooth_window=None, decimate=1,
                             show_algorithm_in_label=True):
    """Helper que plotea la lista de experiments en los ejes provistos.
    - smooth_window: si int>1 aplica media móvil a las series antes de plotear.
    - decimate: dibuja cada "decimate" punto para reducir solapamiento (1 = todos).
    """

    if not experiments:
        return

    # Colores consistentes por algoritmo
    algs = list(OrderedDict.fromkeys([exp.get('parameters', {}).get('algorithm') for exp in experiments]))
    cmap = plt.get_cmap('tab10')
    color_map = {alg: cmap(i % 10) for i, alg in enumerate(algs)}

    for exp in experiments:
        params = exp.get('parameters', {})
        history = exp.get('history', []) or []
        if not history:
            continue

        iters = [h.get('k', i) for i, h in enumerate(history)]
        f_vals = [h.get('f', np.nan) for h in history]
        grad_norms = [h.get('grad_norm', np.nan) for h in history]

        # Etiqueta compacta: por defecto mostramos algoritmo + x0 redondeado.
        # Si show_algorithm_in_label=False (cuando ya agrupamos por algoritmo),
        # sólo mostramos x0 para mantener la leyenda corta.
        x0 = params.get('x0')
        try:
            x0_str = tuple(round(float(v), 3) for v in x0) if hasattr(x0, '__iter__') else round(float(x0), 3)
        except Exception:
            x0_str = str(x0)
        # Mapear nombres de algoritmo a etiquetas más legibles
        alg_display_map = {'gd': 'GD-Armijo', 'bfgs': 'BFGS'}
        alg_display = alg_display_map.get(params.get('algorithm'), params.get('algorithm'))
        if compact_labels:
            if show_algorithm_in_label:
                label = f"{alg_display} {x0_str}"
            else:
                label = f"{x0_str}"
        else:
            if show_algorithm_in_label:
                label = f"{params.get('algorithm')} x0={x0_str} ls={params.get('line_search')}"
            else:
                label = f"x0={x0_str} ls={params.get('line_search')}"

        color = color_map.get(params.get('algorithm'))

        f_vals_arr = np.array(f_vals, dtype=float)
        grad_vals_arr = np.array(grad_norms, dtype=float)

        # Aplicar suavizado simple si se pidió
        def smooth(arr, w):
            if w is None or w <= 1:
                return arr
            kernel = np.ones(w) / float(w)
            return np.convolve(arr, kernel, mode='valid')

        if smooth_window and smooth_window > 1:
            f_vals_s = smooth(f_vals_arr, smooth_window)
            grad_vals_s = smooth(grad_vals_arr, smooth_window)
            its = iters[smooth_window - 1:]
        else:
            f_vals_s = f_vals_arr
            grad_vals_s = grad_vals_arr
            its = iters

        # Decimar para evitar dibujar demasiados puntos
        idx = np.arange(0, len(its), decimate)
        its_d = np.array(its)[idx]
        f_d = f_vals_s[idx]
        g_d = grad_vals_s[idx]

        # f
        if log_scale and np.all(np.isfinite(f_d)) and np.all(f_d > 0):
            ax_f.semilogy(its_d, f_d, marker=None, label=label, alpha=0.8, color=color, linewidth=1.5)
        else:
            ax_f.plot(its_d, f_d, marker=None, label=label, alpha=0.8, color=color, linewidth=1.5)

        # grad norm
        if log_scale and np.all(np.isfinite(g_d)) and np.all(g_d > 0):
            ax_grad.semilogy(its_d, g_d, marker=None, label=label, alpha=0.8, color=color, linewidth=1.5)
        else:
            ax_grad.plot(its_d, g_d, marker=None, label=label, alpha=0.8, color=color, linewidth=1.5)


def plot_convergence(experiments, filename='convergence.png', log_scale=True,
                     legend_outside=False, legend_fontsize='small', compact_labels=True,
                     rect_right=0.85, group_by=None, smooth_window=None, decimate=1):
    """Plotea f(k) y ||grad||(k) para una lista de experimentos.
    - experiments: lista de dicts con 'parameters' y 'history'
    - log_scale: si True intenta usar escala log; si hay valores no positivos usa escala lineal.
    - legend_outside: si True sitúa la leyenda fuera del área de la figura y ajusta el layout.
    - group_by: si 'algorithm' genera una figura separada por cada algoritmo (archivos con sufijo _{alg}).
    - smooth_window: int o None, aplica media móvil para suavizar curvas largas.
    - decimate: int >=1, toma cada "decimate" punto para reducir cantidad de marcadores.
    """

    if not experiments:
        print("No hay experimentos para graficar.")
        return

    if group_by == 'algorithm':
        # generar una figura separada por algoritmo
        algs = list(OrderedDict.fromkeys([exp.get('parameters', {}).get('algorithm') for exp in experiments]))
        for alg in algs:
            subset = [exp for exp in experiments if exp.get('parameters', {}).get('algorithm') == alg]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax_f, ax_grad = axes
            # Cuando ya agrupamos por algoritmo no hace falta repetir el nombre del algoritmo
            # en cada etiqueta de la leyenda; pasamos show_algorithm_in_label=False.
            _plot_convergence_single(subset, ax_f, ax_grad, log_scale=log_scale,
                                     legend_fontsize=legend_fontsize, compact_labels=compact_labels,
                                     smooth_window=smooth_window, decimate=decimate,
                                     show_algorithm_in_label=False)
            ax_f.set_xlabel('Iteración')
            ax_f.set_ylabel('f(x)')
            alg_display_map = {'gd': 'GD-Armijo', 'bfgs': 'BFGS'}
            alg_display = alg_display_map.get(alg, alg)
            ax_f.set_title(f'Convergencia de la función objetivo — {alg_display}')
            ax_grad.set_xlabel('Iteración')
            ax_grad.set_ylabel('‖∇f(x)‖')
            ax_grad.set_title('Convergencia del gradiente')
            for ax in (ax_f, ax_grad):
                ax.grid(True, alpha=0.3)
            if legend_outside:
                ax_f.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
                ax_grad.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
                plt.tight_layout(rect=[0, 0, rect_right, 1])
            else:
                ax_f.legend(fontsize=legend_fontsize)
                ax_grad.legend(fontsize=legend_fontsize)
                plt.tight_layout()
            fname = filename.replace('.png', f'_{alg}.png') if filename.endswith('.png') else f"{filename}_{alg}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada: {fname}")
            plt.show()
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_f, ax_grad = axes
    # En el caso no agrupado mostramos el algoritmo en las etiquetas para identificar cada curva.
    _plot_convergence_single(experiments, ax_f, ax_grad, log_scale=log_scale,
                             legend_fontsize=legend_fontsize, compact_labels=compact_labels,
                             smooth_window=smooth_window, decimate=decimate,
                             show_algorithm_in_label=True)

    ax_f.set_xlabel('Iteración')
    ax_f.set_ylabel('f(x)')
    ax_f.set_title('Convergencia de la función objetivo')
    ax_grad.set_xlabel('Iteración')
    ax_grad.set_ylabel('‖∇f(x)‖')
    ax_grad.set_title('Convergencia del gradiente')
    for ax in (ax_f, ax_grad):
        ax.grid(True, alpha=0.3)
    if legend_outside:
        ax_f.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax_grad.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout(rect=[0, 0, rect_right, 1])
    else:
        ax_f.legend(fontsize=legend_fontsize)
        ax_grad.legend(fontsize=legend_fontsize)
        plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.show()
    if not experiments:
        print("No hay experimentos para graficar.")
        return

    algs = list(OrderedDict.fromkeys([exp.get('parameters', {}).get('algorithm') for exp in experiments]))
    cmap = plt.get_cmap('tab10')
    color_map = {alg: cmap(i % 10) for i, alg in enumerate(algs)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_f, ax_grad = axes
    for exp in experiments:
        params = exp.get('parameters', {})
        history = exp.get('history', []) or []
        if not history:
            continue

        iters = [h.get('k', i) for i, h in enumerate(history)]
        f_vals = [h.get('f', np.nan) for h in history]
        grad_norms = [h.get('grad_norm', np.nan) for h in history]

        x0 = params.get('x0')
        try:
            x0_str = tuple(round(float(v), 3) for v in x0) if hasattr(x0, '__iter__') else round(float(x0), 3)
        except Exception:
            x0_str = str(x0)
        if compact_labels:
            label = f"{params.get('algorithm')} {x0_str}"
        else:
            label = f"{params.get('algorithm')} x0={x0_str} ls={params.get('line_search')}"

        color = color_map.get(params.get('algorithm'))

        f_vals_arr = np.array(f_vals, dtype=float)
        grad_vals_arr = np.array(grad_norms, dtype=float)

        if log_scale and np.all(np.isfinite(f_vals_arr)) and np.all(f_vals_arr > 0):
            ax_f.semilogy(iters, f_vals_arr, marker='o', label=label, alpha=0.9, color=color, linewidth=1.5, markersize=4)
        else:
            if log_scale and np.any(f_vals_arr <= 0):
                print(f"Advertencia: f contiene valores no positivos para {label}. Usando escala lineal.")
            ax_f.plot(iters, f_vals_arr, marker='o', label=label, alpha=0.9, color=color, linewidth=1.5, markersize=4)

        if log_scale and np.all(np.isfinite(grad_vals_arr)) and np.all(grad_vals_arr > 0):
            ax_grad.semilogy(iters, grad_vals_arr, marker='s', label=label, alpha=0.9, color=color, linewidth=1.5, markersize=4)
        else:
            if log_scale and np.any(grad_vals_arr <= 0):
                print(f"Advertencia: grad contiene valores no positivos para {label}. Usando escala lineal.")
            ax_grad.plot(iters, grad_vals_arr, marker='s', label=label, alpha=0.9, color=color, linewidth=1.5, markersize=4)

    ax_f.set_xlabel('Iteración')
    ax_f.set_ylabel('f(x)')
    ax_f.set_title('Convergencia de la función objetivo')
    ax_grad.set_xlabel('Iteración')
    ax_grad.set_ylabel('‖∇f(x)‖')
    ax_grad.set_title('Convergencia del gradiente')
    for ax in (ax_f, ax_grad):
        ax.grid(True, alpha=0.3)

    if legend_outside:
        ax_f.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax_grad.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout(rect=[0, 0, rect_right, 1])
    else:
        ax_f.legend(fontsize=legend_fontsize)
        ax_grad.legend(fontsize=legend_fontsize)
        plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.show()


def plot_final_vs_iters(experiments, filename='final_vs_iters.png', group_by=None):
    points = []
    for exp in experiments:
        res = exp.get('results', {})
        params = exp.get('parameters', {})
        it = res.get('iterations')
        fv = res.get('f_final')
        alg = params.get('algorithm')
        x0 = params.get('x0')
        if it is None or fv is None:
            continue
        try:
            it_n = int(it)
            fv_n = float(fv)
        except Exception:
            continue
        points.append((it_n, fv_n, alg, x0))

    if not points:
        print('No hay datos válidos para plot_final_vs_iters')
        return

    # Si se solicita agrupar por algoritmo, generamos una figura separada por cada algoritmo
    if group_by == 'algorithm':
        algs = list(OrderedDict.fromkeys([p[2] for p in points]))
        for alg in algs:
            subset = [p for p in points if p[2] == alg]
            if not subset:
                continue
            fname = filename.replace('.png', f'_{alg}.png') if filename.endswith('.png') else f"{filename}_{alg}.png"
            plot_final_vs_iters(subset_to_experiments(subset), filename=fname, group_by=None)
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    algs = list(OrderedDict.fromkeys([p[2] for p in points]))
    cmap = plt.get_cmap('tab10')
    color_map = {alg: cmap(i % 10) for i, alg in enumerate(algs)}

    iters_arr = np.array([p[0] for p in points], dtype=float)
    f_arr = np.array([p[1] for p in points], dtype=float)
    algs_arr = [p[2] for p in points]
    x0s = [p[3] for p in points]

    # decidir si usamos escala log en Y: solo si todos los valores son positivos
    # y existe una razón para separar órdenes de magnitud
    use_log = False
    if np.all(np.isfinite(f_arr)) and np.all(f_arr > 0):
        fmax = float(np.max(f_arr))
        fmin = float(np.min(f_arr))
        if fmax / max(fmin, 1e-16) > 50.0:
            use_log = True

    # aplicar jitter horizontal determinista para reducir solapamiento
    x_range = max(1.0, float(np.max(iters_arr) - np.min(iters_arr)))
    jitter_scale = max(1.0, x_range * 0.015)  # 1.5% of range, at least 1 iteration

    # calcular offsets por algoritmo (dodge) para separar claramente GD vs BFGS
    alg_positions = {alg: i for i, alg in enumerate(algs)}
    n_algs = max(1, len(algs))
    # spread algorithms symmetrically around 0
    alg_offset_map = {}
    for alg, idx in alg_positions.items():
        alg_offset_map[alg] = (idx - (n_algs - 1) / 2.0) * (jitter_scale * 1.8)

    jittered_x = np.zeros_like(iters_arr)
    for i, (it_n, fv_n, alg, x0) in enumerate(points):
        # generar semilla estable a partir de algoritmo + x0
        key = f"{alg}|{repr(x0)}"
        h = hashlib.md5(key.encode('utf-8')).hexdigest()
        h_int = int(h[:8], 16)
        # mapear a un valor en [-0.5, 0.5]
        frac = (h_int % 1000) / 1000.0 - 0.5
        small_jitter = frac * (jitter_scale * 0.4)
        alg_offset = alg_offset_map.get(alg, 0.0)
        jittered_x[i] = float(it_n) + alg_offset + small_jitter

    # dibujar puntos por algoritmo (mismo color por algoritmo) usando jitter+dodge
    for alg in algs:
        mask = np.array([a == alg for a in algs_arr])
        if not np.any(mask):
            continue
        ax.scatter(jittered_x[mask], f_arr[mask], color=color_map.get(alg), edgecolor='k', linewidth=0.4, s=60, alpha=0.95, label=alg.upper())

    # Si hay muchos puntos, anotamos sólo los más relevantes; si son pocos, anotamos todos.
    n_points = len(points)
    annotate_all = n_points <= 14

    # construir offsets cíclicos para reducir solapamiento
    offset_cycle = [(6, 6), (-12, 6), (6, -10), (-12, -10), (12, 0), (0, 12), (-8, 0), (0, -12)]

    if annotate_all:
        indices_to_annotate = list(range(n_points))
    else:
        # seleccionar top por f y por iteraciones
        idx_by_f = np.argsort(-f_arr)[:8]
        idx_by_it = np.argsort(-iters_arr)[:8]
        indices_to_annotate = list(dict.fromkeys(list(idx_by_f) + list(idx_by_it)))[:12]

    for i in indices_to_annotate:
        it_n, fv_n, alg, x0 = points[i]
        try:
            x0_str = tuple(round(float(v), 3) for v in x0) if hasattr(x0, '__iter__') else round(float(x0), 3)
        except Exception:
            x0_str = str(x0)
        label = f"{alg.upper()} {x0_str}"
        dx, dy = offset_cycle[i % len(offset_cycle)]
        ax.annotate(label, (it_n, fv_n), xytext=(dx, dy), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7), ha='left', va='bottom')

    ax.set_xlabel('Iteraciones totales')
    ax.set_ylabel('Valor final de f(x)')
    ax.set_title('Comparación: f_final vs iteraciones')
    ax.grid(True, alpha=0.3)
    if use_log:
        ax.set_yscale('log')

    # leyenda para algoritmos
    ax.legend(title='Algoritmo', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.show()


def subset_to_experiments(subset_points):
    """Helper temporal que convierte una lista de tuplas (it, fv, alg, x0)
    a la forma esperada por el código existente: una lista de 'experiments' con
    campos mínimos necesarios para plot_final_vs_iters.
    Esto evita duplicar la lógica de extracción de puntos.
    """
    exps = []
    for it_n, fv_n, alg, x0 in subset_points:
        exps.append({'results': {'iterations': int(it_n), 'f_final': float(fv_n)}, 'parameters': {'algorithm': alg, 'x0': x0}})
    return exps


def plot_trajectory_2d(experiment, filename='trajectory.png', objective_fn=None, x_label='x', y_label='y'):
    history = experiment.get('history', []) or []
    if not history:
        print('Advertencia: history vacío — nada que graficar.')
        return

    xs_raw = [h.get('x') for h in history]
    try:
        xs = np.array(xs_raw, dtype=float)
    except Exception:
        print('Advertencia: no se pudo convertir la trayectoria a float; comprueba los datos de history.')
        return

    if xs.ndim != 2 or xs.shape[1] != 2:
        print('Advertencia: Trayectoria 2D no aplicable (dimensión != 2).')
        return

    margin = 0.5
    x_min, x_max = xs[:, 0].min() - margin, xs[:, 0].max() + margin
    y_min, y_max = xs[:, 1].min() - margin, xs[:, 1].max() + margin
    x_grid = np.linspace(x_min, x_max, 200)
    y_grid = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_grid, y_grid)

    if objective_fn is not None:
        try:
            Z = objective_fn(X, Y)
        except Exception:
            Z = np.vectorize(lambda xx, yy: objective_fn(np.array([xx, yy])))(X, Y)
    else:
        Z = Y**2 + np.log(1.0 + X**2)

    Z = np.nan_to_num(Z, nan=np.nanmean(Z))
    fig, ax = plt.subplots(figsize=(8, 7))
    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(xs[:, 0], xs[:, 1], 'o-', color='C1', linewidth=2, markersize=6, label='Trayectoria')
    ax.scatter(xs[0, 0], xs[0, 1], marker='s', color='C2', s=100, label=f'Inicio x0={list(map(float, xs[0]))}')
    ax.scatter(xs[-1, 0], xs[-1, 1], marker='*', color='C3', s=150, label=f'Final x*={list(map(float, xs[-1]))}')
    ax.scatter(0, 0, marker='x', color='k', s=120, label='Óptimo teórico')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Trayectoria optimizador – Exp {experiment.get('id', '?')}")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.show()
