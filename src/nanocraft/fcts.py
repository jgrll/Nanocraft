import numpy as np
import os

from .utls import resolve_user_path, vec_diff_unit
from .cnfg import compute_default_bl_map, COORD_MAP, NC_TOP_DIR, RAT_MAP
from .ioxyz import get_formula_from_filepath, get_filepath_for_xyz, load_xyz
from .srflyrs import get_core_atoms_and_outer_layer
from .molutls import get_closest_ats_in_mol, get_furthest_ats_in_mol, are_ats_in_mol, is_at_in_mol
from .log_cnfg import logger

global FACET_DICT

def _sanitize_facets_str_list(any_facet): 
    try:
        any_facet = [int(x) for x in list(any_facet)] # no problem if already a list of int
        len(any_facet) == 3
    except (ValueError, TypeError):
        logger.error(f"Invalid facet format: {any_facet}. Expected list of 3 integers or string like 'hkl'.")
    return any_facet


def _get_facet_id(any_input):    
    if isinstance(any_input, int):   # is key
        return FACET_DICT[any_input] # [h, k, l]

    any_input = _sanitize_facets_str_list(any_input) # must be [h, k, l] or "hkl" (or fail there)
    for fid, hkl in FACET_DICT.items():
        if hkl == any_input:
            return fid # key
    raise ValueError(f"Facet {any_input} not found in FACET_DICT")


def get_facets_from_family(any_family):
    any_family = _sanitize_facets_str_list(any_family)
    miller_sum = np.sum(np.abs(any_family))

    family_100 = [
        [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1],
        [-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]
        ]                 
    family_110 = [
        [ 1,  1,  0], [ 1,  0,  1], [ 0,  1,  1],
        [ 1, -1,  0], [ 1,  0, -1], [ 0,  1, -1],
        [-1, -1,  0], [-1,  0, -1], [ 0, -1, -1],
        [-1,  1,  0], [-1,  0,  1], [ 0, -1, 1]
        ]
    family_111 = [
        [ 1,  1,	1], [ 1,  1, -1], [ 1, -1,  1], [-1,  1,  1],
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [ 1, -1, -1]
        ]

    match miller_sum:
        case 1: return family_100
        case 2: return family_110
        case 3: return family_111
        case _: raise ValueError("Wrong family given.")


def is_facet_defined(info_facets, any_facet):
    any_facet = _sanitize_facets_str_list(any_facet)
    facet_id = _get_facet_id(any_facet)
    return np.any(np.isin(info_facets[:, 0:4], facet_id))


def get_ats_from_facet(mol, facets_info, any_facet):
    any_facet = _sanitize_facets_str_list(any_facet)
    facet_id  = _get_facet_id(any_facet)
    facet_ats = []

    for i, row in enumerate(facets_info):
        facet_slots = row[:4]
        if facet_id in facet_slots:
            facet_ats.append(mol[i])

    return np.atleast_2d(np.array(facet_ats, dtype=object))


def get_ats_from_family(mol, facets_info, any_family):
    facets = get_facets_from_family(any_family)
    family_ats = []

    for facet in facets:
        facet_ats = get_ats_from_facet(mol, facets_info, facet)
        if facet_ats.size > 0:
            family_ats.append(facet_ats)

    if not family_ats:
        return np.array([], dtype=object)
    else:
        return np.concatenate(family_ats, axis=0)


def get_sub_facet_info(cluster, facet_info, subset_atoms):
    subset_facet_info = np.empty((len(subset_atoms), 6), dtype=object)
    for i, at in enumerate(subset_atoms):
        for j, cluster_at in enumerate(cluster):
            if np.array_equal(at, cluster_at):
                subset_facet_info[i] = facet_info[j]
                break
    return subset_facet_info


def _find_facets(xyz_data, surface_atoms):
    def _update_step(r, dr):
        mask = np.abs(r) >= 1e-6
        r[mask] -= np.sign(r[mask]) * dr
        return r

    def _scan_facet(all_ats, surf_ats, facets_info, hkl_facet, r_ini, dr, sblabw):
        r = np.array(hkl_facet, dtype=float) * r_ini
        maxstep = int(r_ini / dr)
        found = False
        at_zero = False

        surf_types = COORD_MAP['surf_coord'].keys()
        cutoff = min(RAT_MAP[atom] for atom in surf_types)
        pts = np.array(surf_ats[:, 1:], dtype=float)

        # Scan along facet normal
        while not found and not at_zero:
            norm_fac = 1 / np.sqrt(np.sum(np.abs(hkl_facet)))
            dists = np.abs(np.sum([hkl_facet[i] * (pts[:, i] - r[i]) for i in range(3)], axis=0))
            near_plane = norm_fac * dists <= cutoff

            if np.any(near_plane):
                r = _update_step(r, cutoff)
                found = True
            else:
                r = _update_step(r, dr)
                maxstep -= 1

            if maxstep < 0:
                at_zero = True

        if found:
            norm_fac = 1 / np.sqrt(np.sum(np.abs(hkl_facet)))
            dists = np.abs(np.sum([hkl_facet[i] * (pts[:, i] - r[i]) for i in range(3)], axis=0))
            in_slab = norm_fac * dists <= sblabw
            slab_ats = surf_ats[in_slab]

            facet_id = _get_facet_id(hkl_facet)
            current_facet_population = len(slab_ats)

            for i, at in enumerate(all_ats):
                if are_ats_in_mol(at, slab_ats):
                    slot_found = False
                    for slot in range(4): # find empty slot
                        if facets_info[i, slot] == 0:
                            facets_info[i, slot] = facet_id
                            slot_found = True
                            break
                    if not slot_found: # erase last one if this facet is more populated
                        last_slot_facet_id = facets_info[i, 3]
                        if last_slot_facet_id != 0:
                            last_facet_population = np.sum(facets_info[:, :4] == last_slot_facet_id)
                            if current_facet_population > last_facet_population:
                                facets_info[i, 3] = facet_id

            return facets_info
    # Define scanning parameters
    surf_bonds = compute_default_bl_map(list(COORD_MAP['surf_coord'].keys())) # we want true default
    slab_width = 0.5 * min(surf_bonds.values())
    step_size  = 0.1 * slab_width
    scan_start = np.max(np.abs(surface_atoms[:, 1:]) + 2.0)
    # Initialize facet information array 
    nrows = xyz_data.shape[0]
    facets_info = np.empty((nrows, 6), dtype=object)
    facets_info[:, 0:4] = 0 
    facets_info[:, 5] = xyz_data[:, 0]
    for i, atom in enumerate(xyz_data):
        facets_info[i, 4] = 'surf' if is_at_in_mol(atom, surface_atoms) else 'bulk'    
    # Scan for facets
    families = [
        get_facets_from_family('100'),
        get_facets_from_family('110'),
        get_facets_from_family('111')
    ]

    for family in families:
        for facet in family:
            facets_info = _scan_facet(
                xyz_data, surface_atoms, 
                facets_info, facet, 
                scan_start, step_size, slab_width
            )

    return facets_info


def _rem_illdef_facets(info_facets, mol):
    def _check_facet_size(facet_atoms):
        return len(facet_atoms) >= 3

    def _check_facet_geometry(facet_atoms):
        ref_atom = facet_atoms[0]
        closest_atom  = get_closest_ats_in_mol(facet_atoms, ref_atom, nclosest=1)[0]
        furthest_atom = get_furthest_ats_in_mol(facet_atoms, ref_atom, nfurthest=1)[0]

        u = vec_diff_unit(ref_atom, closest_atom)
        v = vec_diff_unit(ref_atom, furthest_atom)
        dot_product = np.abs(np.dot(u, v))
        return dot_product < 0.95 # True if vectors are not collinear

    def _check_facet_unicity(facet_atoms, facet_id, info_facets):
        all_other_facet_ids = [] # collect all other facet ids for all atoms in the facet
        for atom in facet_atoms:
            for i, mol_atom in enumerate(mol): # get back its position in info_faces
                if np.array_equal(mol_atom, atom):
                    facet_slots = info_facets[i, 0:4]
                    atom_other_facets = [fid for fid in facet_slots if fid not in [0, facet_id]]
                    all_other_facet_ids.extend(atom_other_facets)
                    break

        all_other_facet_set = set(all_other_facet_ids)
        for other_facet in all_other_facet_set:
            count = all_other_facet_ids.count(other_facet)
            if count >= len(facet_atoms): # facet_atoms all belong to this other facet
                return False

        return True  # at least one atom of facets_atoms belongs to a distinct facet than its fellows

    for facet_id, facet in FACET_DICT.items():
        if facet_id == 0:  # skip the "no facet" entry
            continue

        facet_atoms = get_ats_from_facet(mol, info_facets, facet)
        if not (_check_facet_size(facet_atoms) and
                _check_facet_geometry(facet_atoms) and
                _check_facet_unicity(facet_atoms, facet_id, info_facets)
            ): # remove the facet assignment for all atoms
            for row in info_facets:
                for slot in range(4):
                    if row[slot] == facet_id:
                        row[slot] = 0
    return info_facets


def _get_filepath_for_facets(xyz_data, output_dir, ill_defined=True):
    suffix = "illdef_faces" if ill_defined else "faces"
    xyz_file = get_filepath_for_xyz(xyz_data, output_dir, suffix)
    return f'{os.path.splitext(xyz_file)[0]}.dat'


def _save_facets(info_facets, filename):
    with open(filename, 'w') as file:
        for row in info_facets:
            facet_slots = '   '.join(f"{int(row[i]):2}" for i in range(4))  # Slots 0-3
            location    = row[4]
            atom_label  = row[5]
            formatted_line = f"{facet_slots}   {location}   {atom_label}"
            file.write(f"{formatted_line}\n")
    return


def generate_facets(data, output_dir=None):
    try:
        xyz_data = load_xyz(data, center_COM=True) if isinstance(data, str) else data.copy()
    except Exception as e:
        logger.error(f"Error loading XYZ data: {e}")
        raise

    _, outer_layer = get_core_atoms_and_outer_layer(xyz_data, 'cap')

    # working only with a bare cluster, all core elements must have coordination matching to the JSON
    if len(outer_layer) > 0:
        logger.error("Stop due to ligands detected in the input data.")
        raise AttributeError("Input data should be a bare nanocrystal (no passivation). Capping ligands were detected.")
    
    _, outer_layer = get_core_atoms_and_outer_layer(xyz_data, 'nc')
    facets_ill = _find_facets(xyz_data, outer_layer)
    facets     = _rem_illdef_facets(facets_ill.copy(), xyz_data)

    if output_dir:
        pass
    elif isinstance(data, np.ndarray):
        output_dir = os.path.join(NC_TOP_DIR, 'facets')
    else:
        output_dir = os.path.join(os.path.dirname(data), 'facets')
    
    filepath_ill = _get_filepath_for_facets(xyz_data, output_dir, ill_defined=True)
    filepath     = _get_filepath_for_facets(xyz_data, output_dir, ill_defined=False)
    resolve_user_path(filepath_ill, 'w') # make sure directories exist

    _save_facets(facets, filepath)
    _save_facets(facets_ill, filepath_ill)

    return facets, facets_ill


def load_facets(xyz_filepath):
    def _load(filepath_xyz, filepath_facet):
        info_facets = np.loadtxt(filepath_facet, dtype=object)
        info_facets[:, :4] = info_facets[:, :4].astype(int)  # 4 facet IDs
        info_facets[:, 4:] = info_facets[:, 4:].astype(str)  # location + atom_label

        xyz_data = load_xyz(filepath_xyz, center_COM=False)

        if xyz_data.shape[0] < info_facets.shape[0]:
            raise ValueError("The xyz file should at least have as many atoms as the dat file.")
        
        for i in range(len(xyz_data)):
            if xyz_data[i, 0] != info_facets[i, 5]:
                raise ValueError("Atom order mismatch between dat and xyz files.")

        return info_facets

    resolve_user_path(xyz_filepath, 'r')
    cluster_formula = get_formula_from_filepath(xyz_filepath)
    base_faces_dat = f"{cluster_formula}_faces.dat"
    base_illdef_faces_dat = f"{cluster_formula}_illdef_faces.dat"
    
    base_dir = os.path.dirname(xyz_filepath)
    search_dirs = [os.path.join(base_dir, 'facets'), os.path.join(base_dir, '../facets')]
    
    found = False
    for dirpath in search_dirs:
        faces_dat        = os.path.join(dirpath, base_faces_dat)
        illdef_faces_dat = os.path.join(dirpath, base_illdef_faces_dat)
        
        if os.path.exists(faces_dat) and os.path.exists(illdef_faces_dat):
            info_facets         = _load(xyz_filepath, faces_dat)
            info_facets_illdef  = _load(xyz_filepath, illdef_faces_dat)
            found = True
            continue

    if not found:
        info_facets, info_facets_illdef = generate_facets(xyz_filepath)
        logger.info(f"No facet files found associated with {xyz_filepath}. Were generated again.")

    return info_facets, info_facets_illdef


def _init_facet_dict():
    global FACET_DICT
    FACET_DICT = {}

    families = [
        get_facets_from_family('100'), 
        get_facets_from_family('110'), 
        get_facets_from_family('111')
        ]

    facet_counter = 1    
    for family in families:
        for facet in family:
            FACET_DICT[facet_counter] = facet
            facet_counter += 1
    return

_init_facet_dict() # init on first called of the module
