import numpy as np
import glob
import re
import os

from .log_cnfg import logger
from .utls import resolve_user_path, compute_COM
from .cnfg import LABEL_SET_ORDERED, COORD_MAP_ATOMS


def save_lines(lines, filepath):
    with open(filepath, "w") as out:
        out.writelines(lines)
    logger.info(f"File saved to {filepath} !")
    return


def check_coord_map(atom_list):
    atom_set = set(atom_list)
    missing = atom_set - set(COORD_MAP_ATOMS)
    if not missing:
        return True
    logger.error(f"Atom types {missing} not found in coordination map.")
    return False


def sort_label_list(label_list):
    return sorted(label_list, key=lambda x: LABEL_SET_ORDERED.get(x, float("inf")))


def get_formula_from_xyz(xyz_data):
    labels = xyz_data[:, 0]
    labels_unique = np.unique(labels)
    labels_sorted = sort_label_list(labels_unique)
    labels_listed = labels.tolist()
    return "".join(label + str(labels_listed.count(label)) for label in labels_sorted)


def get_formula_from_list(labels_listed):
    labels_unique = np.unique(labels_listed)
    labels_sorted = sort_label_list(labels_unique)
    return "".join(label + str(labels_listed.count(label)) for label in labels_sorted)


def get_dict_from_formula(formula: str): 
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    return {element: int(count) if count else 1 for element, count in matches}


def get_anchor_from_formula(formula):
    split = list(formula)
    if len(split) == 0:
        return str('')
    elif len(split) == 1:
        return split[0]
    else:
        guess = split[0]
        if not isinstance(split[1], int) and split[1].islower():
            guess += split[1]
        return guess
    

def get_xyz_header(xyz_data, header=None):
    if not header: # default
        header = " ".join(sort_label_list(np.unique(xyz_data[:, 0])))
    return f"{len(xyz_data)}\n{header}\n" # xyz two lines header


def get_filepath_for_xyz(xyz_data, output_dir, suffix=None) -> str:
    prefix = get_formula_from_xyz(xyz_data)
    filename = f"{prefix}_{suffix}.xyz" if suffix else f"{prefix}.xyz"
    return os.path.join(output_dir, filename)


def _read_num_atoms(inputfile):
    with open(inputfile, 'r') as f:
        first_line = f.readline().strip()
    try:
        return int(first_line)
    except ValueError as e:
        raise ValueError("First line of XYZ file is not an integer.") from e


def _validate_xyz(inputfile, data):
    num_atoms = _read_num_atoms(inputfile)
    if num_atoms != data.shape[0]:
        message = f"XYZ atom count mismatch: expected {num_atoms}, got {data.shape[0]}"
        logger.error(message)
        raise ValueError(message)
    
    if data.shape[1] not in (3, 4):
        message = f"Invalid columns in XYZ: expected 3 or 4, got {data.shape[1]}"
        logger.error(message)
        raise ValueError(message)

    atom_types = np.unique(data[:, 0])
    if not check_coord_map(atom_types.tolist()):
        message = f"XYZ contains unsupported atom types: {atom_types.tolist()}"
        logger.error(message)
        raise ValueError(message)

    logger.info("XYZ validation passed: all atom types found in coordination map.")


def load_xyz(inputfile: str, center_COM=False, sanity_check=True) -> np.ndarray:
    resolve_user_path(inputfile, 'r')
    logger.info(f"Loading: {inputfile}")
    data = np.atleast_2d(np.loadtxt(inputfile, skiprows=2, dtype=object))
    data[:, 1:] = data[:, 1:].astype(float)
    
    if sanity_check:
        _validate_xyz(inputfile, data)

    if center_COM:
        data[:, 1:] -= compute_COM(data)
        logger.info("Structure's COM set to (0.0, 0.0, 0.0).")
    
    logger.info(f"{inputfile} loaded successfully.")
    return data


def sort_by_postfix_int(filepaths):
    sorted_files = []
    unsorted_files = []

    for filepath in filepaths:
        if not filepath.lower().endswith('.xyz'):
            unsorted_files.append(filepath)
            continue

        basename = os.path.basename(filepath)
        if match := re.search(r'_(\d+)\.xyz$', basename):
            number = int(match[1])
            sorted_files.append((number, filepath))

        else:
            unsorted_files.append(filepath)

    sorted_files.sort(key=lambda x: x[0])
    sorted_filepaths = [filepath for _, filepath in sorted_files]
    return unsorted_files + sorted_filepaths

    
def update_postfix_int(filepath, previous_filepath):
    def _add(filepath, counter):
        basename_wo_ext, ext = os.path.splitext(os.path.basename(filepath))
        new_basename = f"{basename_wo_ext}_{counter}{ext}"
        return os.path.join(os.path.dirname(filepath), new_basename)

    match = re.search(r'_(\d+)$', os.path.splitext(previous_filepath)[0])
    if match: # if previous file already has a postfix int, we increment it
        filepath_updated = _add(filepath, int(match[1]) + 1)

    else:
        default_counter = 1
        previous_filepath_updated = _add(previous_filepath, 1)
        os.rename(previous_filepath, previous_filepath_updated)
        filepath_updated = _add(filepath, default_counter + 1)
    return filepath_updated


def get_formula_from_filepath(filepath):
    pattern = re.compile(r'^((?:[A-Z][a-z]?\d+)+)')
    if match := pattern.match(os.path.basename(filepath)):
        return match[1]
    return None


def _fast_formula_match_check(xyz_data, file_to_check): 
    # fast check if raw formula of both file are equal based on filename only
    xyz_formula = get_formula_from_xyz(xyz_data)
    file_formula = get_formula_from_filepath(file_to_check)
    if file_formula is None:
        return False
    xyz_dict = get_dict_from_formula(xyz_formula)
    match_dict = get_dict_from_formula(file_formula)
    return match_dict == xyz_dict


def _long_formula_match_check(xyz_data, file_to_check):
    # longer check based on the order of labels in the files. The position are not checked
    xyz_check = load_xyz(file_to_check, sanity_check=False, center_COM=False)
    return np.all(xyz_data[:, 0] == xyz_check[:, 0]) if xyz_data.shape[0] == xyz_check.shape[0] else False


def _write_xyz(xyz_data, filepath, header_message=None):
    # save with no check; can be used instead of save_xyz for testing
    header = get_xyz_header(xyz_data, header_message)
    with open(filepath, "w") as f:
        f.write(header)
        for at in xyz_data:
            formatted_line = "{:<2}{:18.10f}{:18.10f}{:18.10f}\n".format(
                at[0], float(at[1]), float(at[2]), float(at[3]))
            f.write(formatted_line)
    logger.info(f"Structure was written to {filepath}.")


def _search_duplicates(xyz_data, filepath):
    checklist = glob.glob(os.path.join(os.path.dirname(filepath), "*.xyz"))
    files_first_check_passed, files_second_check_passed = [], []
    index = 0
    while index < len(checklist):
        filepath_check = checklist[index]
        first_check = _fast_formula_match_check(xyz_data, filepath_check)
        second_check = _long_formula_match_check(xyz_data, filepath_check) if first_check else False
        
        if first_check:
            files_first_check_passed.append(filepath_check)
        if second_check:
            files_second_check_passed.append(filepath_check)
        index += 1
    return files_first_check_passed, files_second_check_passed


def save_xyz(xyz_data, filepath, header_message=None, remove_duplicates=False):
    resolve_user_path(filepath, "w")
    files_matching_formula, files_matching_labels = _search_duplicates(xyz_data, filepath)
    # for duplicates involving ligands, _long_formula_match_check will almost always be False
    # if only formula matches but not atomic labels, we save (postfix needs update)
    if remove_duplicates and len(files_matching_labels) > 0:
        logger.info(f"Structure already exists in {files_matching_labels[0]}, not saved.")
    elif len(files_matching_formula) > 0:
        logger.info("Formula with same formula found, but not same atomic order. Saving...")
        sorted_duplicates = sort_by_postfix_int(files_matching_formula)
        updated_filepath = update_postfix_int(filepath, sorted_duplicates[-1])
        _write_xyz(xyz_data, updated_filepath, header_message)
    else: # no remove_duplicates, or first file of its kind
        _write_xyz(xyz_data, filepath, header_message)
