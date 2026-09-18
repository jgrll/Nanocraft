import os
import numpy as np
from .log_cnfg import logger

def settings_keys_check(settings, valid_keys, mandatory_keys):
    for key in settings.keys():
        if key not in valid_keys:
            logger.warning(f"Warning: Unknown key '{key}' in settings. This key will be ignored.")
    for key in mandatory_keys:
        if key not in settings.keys():
            logger.error(f"Error: Missing key '{key}' in settings. Please provide this key.")
            raise KeyError(f"Error: Missing key '{key}' in settings. Please provide this key.")
    return


def all_same_type(items, target_type):
    if not isinstance(items, list):
        return all_same_type([items], target_type)
    return all(isinstance(item, target_type) for item in items)


def adapt_list_to_length(items, target_length):
    if not isinstance(items, list):
        return adapt_list_to_length([items], target_length)
    if len(items) < 1:
        logger.error("Error: The provided list is empty. Please provide a non-empty list or a single value.")
        raise ValueError("Error: The provided list is empty. Please provide a non-empty list or a single value.")
    if len(items) != target_length:
        logger.warning(f"The list {items} has not the expected length ({target_length}). First element will be duplicated.")
        return [items[0]] * target_length
    return items


def resolve_user_path(path_to_dir_or_file, option):
    if option not in ('r', 'w'):
        raise ValueError(f"Invalid option '{option}', expected 'r' or 'w'.")

    path_to_dir_or_file = os.path.expanduser(os.path.expandvars(path_to_dir_or_file))
    if path_to_dir_or_file.startswith("/"):
        abs_path = os.path.abspath(path_to_dir_or_file)
    else:
        base = os.getcwd()
        abs_path = os.path.abspath(os.path.join(base, path_to_dir_or_file))

    if option == 'r':
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {abs_path}")

    elif option == 'w':
        if os.path.splitext(abs_path)[1] == "":
            os.makedirs(abs_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


def grid(params, collapse=None, duplicate=1):
    if not params:
        return [{}]

    processed_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            processed_params[key] = value.tolist()
        elif not isinstance(value, (list, tuple)):
            processed_params[key] = [value]
        else:
            processed_params[key] = list(value)

    param_names = list(processed_params.keys())
    combinations = [{}]

    for name in param_names:
        values_for_param = processed_params[name]
        expanded_combinations = []

        for existing_combo in combinations:
            for value in values_for_param:
                new_combo = existing_combo.copy()
                new_combo[name] = value
                expanded_combinations.append(new_combo)
        combinations = expanded_combinations

    if collapse:
        param_to_check, param_to_collapse, trigger_values = collapse
        grouped = {}
        for combo in combinations:
            group_key = []
            for k, v in combo.items():
                if k != param_to_collapse:
                    if isinstance(v, (list, np.ndarray)):
                        group_key.append((k, tuple(v)))
                    else:
                        group_key.append((k, v))
            group_key = tuple(group_key)

            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(combo)

        result = []
        for group_combos in grouped.values():
            should_collapse = any(
                combo.get(param_to_check) in trigger_values
                for combo in group_combos
            )
            if should_collapse:
                result.append(group_combos[0])
            else:
                result.extend(group_combos)

        combinations = result

    duplicate = max(1, int(duplicate))  # Ensure at least 1

    return combinations if duplicate == 1 else combinations * duplicate


def run_parallel(func, setting_grid, n_workers=1, **kwargs):
    if n_workers <= 1:
        logger.info("Running sequentially.")
        for setting in tqdm(setting_grid, total=len(setting_grid), desc="Progress"):
            func(setting, **kwargs)
        return
    try:
        from multiprocessing import Pool
        from functools import partial
        from tqdm import tqdm
        partial_func = partial(func, **kwargs)
        with Pool(processes=n_workers) as pool:
            mapped = pool.imap(partial_func, setting_grid)
            for _ in tqdm(mapped, total=len(setting_grid), desc="Progress"):
                pass
    except ImportError:
        logger.warning("Failed to run in parallel. Will run sequentially.")
        run_parallel(func, setting_grid, 1, **kwargs)
    return

def aggregate_dicts(*dicts):
    if not dicts:
        logger.warning("No dictionaries provided to aggregate_coord_dicts")
        return {}

    logger.debug("Dictionary aggregation:")
    for d in dicts: 
        logger.debug(f"{d}")
        for key, value in d.items():
            if isinstance(value, dict):
                logger.error("Key value must not be another dict.")
                raise ValueError("Dictionary values not allowed.")

    result = {}
    all_keys = set()

    for d in dicts:
        all_keys |= set(d.keys())

    for key in all_keys:
        values = []
        for d in dicts:
            if key in d:
                if isinstance(d[key], int):
                    values.append(d[key])
                else:
                    values.extend(d[key])

        result[key] = sorted(set(values))
        logger.debug(f"Aggregated {key}: {result[key]}")

    logger.debug(f"Output aggregated dictionary: {result}")
    return result


def handle_random_seed(seed):
    if isinstance(seed, bool):
        return np.random.randint(0, 2**32 - 1) if seed else 51
    elif isinstance(seed, int):
        return seed
    raise TypeError(f"seed must be bool or int, got {type(seed).__name__}")


def to_unit_interval(value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"Input must be a number, got {type(value).__name__}")
    num = float(value)
    result = num / 100.0 if num > 1.0 else num # choose format
    if result < 0:
        return 0.0
    elif result > 1:
        return 1.0
    else:
        return result
    

def bohr2angstrom(L):
    return L * 0.529177249


def angstrom2bohr(L):
    return L * 1.8897259886


def normalize_point_like(arr):
    arr = np.array(arr, dtype=object)
    if arr.ndim != 1:
        raise ValueError("Point must be a 1D-array.")
    if arr.shape[0] == 3:
        coords = arr
    elif arr.shape[0] == 4:
        coords = arr[1:4]
    else:
        raise ValueError("Point [x,y,z] or [label,x,y,z].")
    coords = np.array(coords, dtype=float)
    return coords


def normalize_data_like(arr):
    arr = np.array(arr, dtype=object)
    if arr.ndim == 1:  # vector (atom)
        if arr.shape[0] == 3:
            coords = np.array(arr, dtype=float)
        elif arr.shape[0] == 4:
            coords = np.array(arr[1:4], dtype=float)
        else:
            raise ValueError("Data as [x,y,z] or [label,x,y,z].")
        coords = coords.reshape(1, 3) # array([[x, y, z]])
        return coords, 1
    elif arr.ndim == 2: # matrix (molecule)
        if arr.shape[1] == 3:
            coords = np.array(arr, dtype=float)
        elif arr.shape[1] == 4:
            coords = np.array(arr[:, 1:4], dtype=float)
        else:
            raise ValueError("Data rows as [x,y,z] or [label,x,y,z].")
        return coords, 2
    else:
        raise ValueError("Data must be a 1D or 2D-array.")


def spheric2cart(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return x, y, z


def get_random_angle(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(0.0, 2.0 * np.pi)


def compute_COM(arr):
    arr, _ = normalize_data_like(arr)
    return np.mean(arr, axis=0)


def vec_diff_unit(u, v):
    u, v = normalize_point_like(u), normalize_point_like(v)
    diff = v - u
    norm = np.linalg.norm(diff)
    if norm == 0:
        raise ValueError("Cannot compute unit vector difference: points are identical.")
    return diff / norm


def complete_set_unit_vec_2(e_para, e_perps):
    e_perps_augmented = np.zeros((4, 3), dtype = float)
    e_perps_augmented[0] = e_perps[0]
    e_perps_augmented[1] = e_perps[1]
    e_perps_augmented[2] = np.cross(e_para, e_perps[0])
    e_perps_augmented[3] = np.cross(e_para, e_perps[1])
    return e_perps_augmented


def complete_set_unit_vec_3(e_para, e_perps):
    e_perps_augmented = np.zeros((6, 3), dtype = float)
    e_perps_augmented[0] = e_perps[0]
    e_perps_augmented[1] = e_perps[1]
    e_perps_augmented[2] = e_perps[2]
    e_perps_augmented[3] = np.cross( np.cross(e_perps[0], e_para), e_para)
    e_perps_augmented[4] = np.cross( np.cross(e_perps[1], e_para), e_para)
    e_perps_augmented[5] = np.cross( np.cross(e_perps[2], e_para), e_para)
    return e_perps_augmented


def rodrigues_rot(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1 - np.cos(angle))
    )


def quick_rot_mat(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        if c == -1:  # anti-parallel case
            return -np.eye(3)
        else:
            return np.eye(3) + vx + np.dot(vx, vx) * (1 / (1 + c))


def arbitrary_rot_mat(arbitratry_vec, theta):
    ux, uy, uz = arbitratry_vec
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    return np.array(
        [
            [
                ux**2 * one_minus_cos + cos_theta,
                ux * uy * one_minus_cos - uz * sin_theta,
                ux * uz * one_minus_cos + uy * sin_theta,
            ],
            [
                ux * uy * one_minus_cos + uz * sin_theta,
                uy**2 * one_minus_cos + cos_theta,
                uy * uz * one_minus_cos - ux * sin_theta,
            ],
            [
                ux * uz * one_minus_cos - uy * sin_theta,
                uy * uz * one_minus_cos + ux * sin_theta,
                uz**2 * one_minus_cos + cos_theta,
            ],
        ]
    )


def rotate_around_axis(xyz_data, axis, theta):
    M = arbitrary_rot_mat(axis, theta)
    if xyz_data.shape[1] > 3:
        at_types = xyz_data[:, 0]
        xyz_work = xyz_data[:, 1:].astype(float)
        rotated = np.dot(xyz_work, M.T)
        return np.column_stack((at_types, rotated))
    elif xyz_data.shape[1] == 3:
        xyz_work = xyz_data.astype(float)
        return np.dot(xyz_work, M.T)
    else:
        raise ValueError("Input array must have shape (N,3) or (N,4)")
