import os
import json

global NC_TOP_DIR
NC_TOP_DIR = os.path.abspath(os.path.join(__file__, '../../..'))


def _top_path(filename: str) -> str:
    return os.path.join(NC_TOP_DIR, filename)


def _load_json(filename: str) -> dict:
    with open(filename, "r") as f:
        data = json.load(f)

    data = {k: v for k, v in data.items() if not k.startswith("_comment")}
    return data


def _load_dat(filename: str, use_cols: list[int], column_types: list[callable]) -> dict:
    if len(use_cols) != len(column_types):
        raise ValueError("use_cols and column_types must have same length.")

    result = {}
    with open(filename) as f:
        for n, line in enumerate(f, 1):

            if not line or line.startswith('#'):
                continue

            parts = line.strip().split()
            
            # load column
            try:
                converted = [t(parts[i]) for i, t in zip(use_cols, column_types)]
            except (IndexError, ValueError, TypeError) as e:
                raise ValueError(f"Line {n}: {e}") from e

            # add to dictionary
            key = (
                converted[0] if len(converted) == 1 else
                converted[0] if len(converted) == 2 else
                frozenset(map(str, converted[:-1]))
            )
            value = (
                converted[0] if len(converted) == 1 else
                converted[-1]
            )

            if key in result:
                raise ValueError(f"Duplicate key {key} at line {n}")
            result[key] = value

    return result


def _sanitize_coord_map(data: dict) -> dict:
        cleaned_all = {}
        for section, atoms in data.items():
            if not isinstance(atoms, dict): # missing value for a dictionary entry
                raise TypeError(f"Section '{section}' should be a dictionary of atom: list[int], even if empty.")

            cleaned_section = {}
            for atom, values in atoms.items():
                if isinstance(values, int):
                    values = [values] # if a single int is given, wrap it in a list     
                elif not (isinstance(values, list) and all(isinstance(x, int) for x in values)):
                    raise TypeError(
                        f"Invalid type for '{atom}' in section '{section}': "
                        f"expected int or list of int, got {type(values).__name__}"
                    ) # ensure it's a list of integers
                cleaned_section[atom] = values

            cleaned_all[section] = cleaned_section
        return cleaned_all


def _initialize_globals():
    global RCOV_MAP, RAT_MAP, EN_MAP, BL_MAP, CUSTOM_BA_MAP
    global LABEL_SET_ORDERED, COORD_MAP, COORD_MAP_ATOMS, AMU_MAP
    global DEFAULT_ANGLE
    global OXBARE_MAP

    print("Initializing nanocraft configuration...")
    custom_bonds_file      = _top_path("misc/bonds.dat")
    custom_angles_file     = _top_path("misc/angles.dat")
    ref_atomic_data_file   = _top_path("misc/atomic.dat")
    ref_label_order_file   = _top_path("misc/order.dat")
    user_coordination_file = _top_path("misc/coordination.json")

    DEFAULT_ANGLE = 104.45 # default bond angle in degrees (water angle)

    RCOV_MAP = _load_dat(ref_atomic_data_file, use_cols=[1,4], column_types=[str, float])
    RCOV_MAP = {k: v / 100.0 for k, v in RCOV_MAP.items()}  # convert from pm to angstrom

    RAT_MAP = _load_dat(ref_atomic_data_file, use_cols=[1,5], column_types=[str, float])
    RAT_MAP = {k: v / 100.0 for k, v in RAT_MAP.items()}   # convert from pm to angstrom

    EN_MAP = _load_dat(ref_atomic_data_file, use_cols=[1,3], column_types=[str, float])

    AMU_MAP = _load_dat(ref_atomic_data_file, use_cols=[1,2], column_types=[str, float])
    
    OXBARE_MAP = _load_dat(ref_atomic_data_file, use_cols=[1,7], column_types=[str, float])

    label_list_ordered = [tpl[0] for tpl in _load_dat(ref_label_order_file, use_cols=[0], column_types=[str]).items()]
    LABEL_SET_ORDERED  = {element: index for index, element in enumerate(label_list_ordered)}

    COORD_MAP       = _sanitize_coord_map(_load_json(user_coordination_file))
    COORD_MAP_ATOMS = list({atom for smap in COORD_MAP.values() for atom in smap.keys()}) # all atoms present in COORD_MAP
    
    custom_BL_MAP  = _load_dat(custom_bonds_file, use_cols=[0,1,2], column_types=[str, str, float])
    default_BL_MAP = compute_default_bl_map(COORD_MAP_ATOMS)
    BL_MAP         = {**default_BL_MAP, **custom_BL_MAP} # custom bond lengths overwrite default ones

    CUSTOM_BA_MAP = _load_dat(custom_angles_file, use_cols=[0,1,2,3], column_types=[str, str, str, float])
    print("Done! If any changes are made to the nanocraft misc files, please restart the environment.")
    return

def compute_default_bl_map(atom_list: list[str]) -> dict:    
    combinations_dict = {}
    n = len(atom_list)

    for i in range(n):
        for j in range(i, n): # j = i to get X-X bonds
            
            atom1, atom2 = atom_list[i], atom_list[j]
            
            bl = RCOV_MAP.get(atom1) + RCOV_MAP.get(atom2)
            en_corr = 0.09 * abs(EN_MAP.get(atom1) - EN_MAP.get(atom2))

            combinations_dict[frozenset((atom1, atom2))] = bl - en_corr

    return combinations_dict

_initialize_globals()
