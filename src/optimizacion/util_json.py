import json
import os
import numpy as np
import sys
import time

def to_python_types(obj):
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, (bool, str, type(None), int, float)):
		return obj
	elif isinstance(obj, dict):
		return {k: to_python_types(v) for k, v in obj.items()}
	elif isinstance(obj, list):
		return [to_python_types(v) for v in obj]
	else:
		return str(obj)

def serialize_experiment_to_json(data: dict, filename: str):
	converted = to_python_types(data)
	dirpath = os.path.dirname(filename)
	if dirpath:
		os.makedirs(dirpath, exist_ok=True)
	with open(filename, 'w', encoding='utf-8') as f:
		json.dump(converted, f, indent=2, ensure_ascii=False)
	print(f"Resultados guardados en: {filename}")
	try:
		print(f"  Tamaño del archivo: {len(json.dumps(converted))/1024:.2f} KB")
	except Exception:
		pass

def load_experiments_from_json(filename: str) -> dict:
	with open(filename, 'r', encoding='utf-8') as f:
		data = json.load(f)
	print(f"Cargado: {filename}")
	print(f"  Metadatos: {list(data.get('metadata', {}).keys())}")
	print(f"  Número de experimentos: {len(data.get('experiments', []))}")
	return data

def create_metadata(objective_function: str, algorithms: list, author: str) -> dict:
	try:
		import scipy
		scipy_version = scipy.__version__
	except ImportError:
		scipy_version = None
	try:
		import matplotlib
		matplotlib_version = matplotlib.__version__
	except ImportError:
		matplotlib_version = None

	metadata = {
		"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
		"schema_version": "1.0",
		"python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
		"numpy_version": np.__version__,
		"scipy_version": scipy_version,
		"matplotlib_version": matplotlib_version,
		"objective_function": objective_function,
		"algorithms": algorithms,
		"author": author
	}
	return metadata

def run_and_save_experiments(run_configs: list, algorithm_fn_map: dict, filename: str, f, grad, mesh_metadata=None):
	"""
	Ejecuta experimentos y guarda resultados en JSON.
	
	Parámetros adicionales:
	-----------------------
	mesh_metadata : dict, opcional
		Diccionario que mapea índices de puntos a metadata de malla.
		Formato: {point_idx: {'mesh_id': int, 'mesh_name': str, 'distance_to_opt': float}}
	"""
	experiment_data = {
		"metadata": create_metadata(
			objective_function="f(x,y) = y^2 + log(1 + x^2)",
			algorithms=list(algorithm_fn_map.keys()),
			author="Tu Nombre"
		),
		"experiments": []
	}

	for idx, cfg in enumerate(run_configs, start=1):
	
		x0_print = np.asarray(cfg['x0']).tolist()
		print(f"\n[Exp {idx}] algoritmo={cfg['algorithm']} x0={x0_print} tol={cfg['tolerance']} ls={cfg.get('line_search')}")
		t0 = time.time()
		result = algorithm_fn_map[cfg["algorithm"]](
			f=f,
			grad=grad,
			x0=cfg["x0"],
			tol=cfg["tolerance"],
			line_search=cfg.get("line_search"),
			maxiter=cfg.get("maxiter", 200)
		)
		t1 = time.time()
		time_elapsed = t1 - t0

		x0_param = np.asarray(cfg.get("x0")).tolist()
		params = {
			"algorithm": cfg.get("algorithm"),
			"x0": x0_param,
			"tolerance": cfg.get("tolerance"),
			"line_search": cfg.get("line_search"),
			"maxiter": cfg.get("maxiter", None)
		}
		
		# Agregar metadata de malla si está disponible
		if mesh_metadata is not None and cfg.get('point_idx') is not None:
			point_idx = cfg.get('point_idx')
			if point_idx in mesh_metadata:
				params['mesh_id'] = mesh_metadata[point_idx]['mesh_id']
				params['mesh_name'] = mesh_metadata[point_idx]['mesh_name']
				params['distance_to_opt'] = mesh_metadata[point_idx]['distance_to_opt']
	
		x_val = result.get("x", None)
		if x_val is None:
			x_final = None
		else:
			x_final = np.asarray(x_val).tolist()

			
			if result.get("grad_norm") is not None:
				grad_norm = float(result.get("grad_norm"))
			else:
				grad_val = result.get("grad", None)
				if grad_val is not None:
					try:
						grad_norm = float(np.linalg.norm(np.asarray(grad_val)))
					except Exception:
						grad_norm = None
				else:
					
					last_hist = result.get("history", [])
					if last_hist:
						last = last_hist[-1]
						grad_norm = float(last.get("grad_norm")) if last.get("grad_norm") is not None else None
					else:
						grad_norm = None

		f_val = result.get("f", None)
		f_final = float(f_val) if f_val is not None else None

		results = {
			"x_final": x_final,
			"f_final": f_final,
			"grad_norm_final": grad_norm,
			"status": result.get("status"),
			"iterations": len(result.get("history", [])),
			"time_seconds": time_elapsed,
			"f_evaluations": result.get("f_evals", None),
			"grad_evaluations": result.get("grad_evals", None)
		}
		history = []
		for h in result.get("history", []):
			x_h = np.asarray(h.get("x")).tolist()
			entry = {
				"k": h.get("k"),
				"x": x_h,
				"f": float(h.get("f")),
				"grad_norm": float(h.get("grad_norm")) if h.get("grad_norm") is not None else None,
				"alpha": h.get("alpha", None),
				"f_evals": h.get("f_evals", None),
				"grad_evals": h.get("grad_evals", None)
			}
			history.append(entry)

		exp_entry = {
			"id": idx,
			"parameters": params,
			"results": results,
			"history": history
		}

		experiment_data["experiments"].append(exp_entry)

	serialize_experiment_to_json(experiment_data, filename)
	return experiment_data

