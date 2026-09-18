import numpy as np
from .cnfg import BL_MAP, CUSTOM_BA_MAP, DEFAULT_ANGLE
from scipy.spatial.distance import cdist
from .utls import quick_rot_mat, arbitrary_rot_mat, normalize_point_like, normalize_data_like

def get_bl_from_map(atom1, atom2):
    key = frozenset((atom1, atom2))
    return float(BL_MAP.get(key, 0.0))


def get_ba_from_map(atom1, atom2, atom3):
    key = frozenset((atom1, atom2, atom3))
    return float(CUSTOM_BA_MAP.get(key, DEFAULT_ANGLE)) # default is water angle between bonds


def distance_between(point_or_at, point_or_at_or_mol):
    point = normalize_point_like(point_or_at) # always (3)
    data, original_ndim = normalize_data_like(point_or_at_or_mol) # always (N,3)
    differences = data - point
    distances = np.linalg.norm(differences, axis=1)
    return distances[0] if original_ndim == 1 else distances


def are_mols_equal(mol1, mol2, threshold=1e-3):
    mol1, mol2 = np.array(mol1, dtype=object), np.array(mol2, dtype=object)
    if mol1.shape[0] != mol2.shape[0]:
        return False
    return all(is_at_in_mol(at1, mol2, threshold=threshold) for at1 in mol1)


def get_ats_ids(atoms, cluster):
    # know for sure that the atom is in the cluster
    # allow to select only closest match(es)
    atoms = np.atleast_2d(atoms)
    return [int(np.argmin(distance_between(atom, cluster))) for atom in atoms]


def atoms_with_n_neighbors(mol, n, filter_type=None):
    mol = np.array(mol, dtype=object)
    indices = []
    for i, at in enumerate(mol):
        _, coord = get_neighbors(at, mol)
        if coord == n:
            indices.append(i)
    result = mol[indices]
    if filter_type is not None:
        result = result[result[:, 0] == filter_type]
    return result


def get_neighbors(at, mol, custom_cutoff=None, tolerance=0.2):
    at = np.array(at, dtype=object)
    mol = np.array(mol, dtype=object)

    if custom_cutoff is not None:
        lookup_cutoff = custom_cutoff
    else:
        lookup_cutoff = get_bl_from_map('Ch', 'Ch') # 0.0 if missing
        lookup_cutoff = 3.050 if lookup_cutoff < 1.00 else lookup_cutoff # safety

    mask = distance_between(at, mol) <= lookup_cutoff
    sphere_around = mol[mask] # only atoms in a sphere of radius lookup_cutoff
    sphere_around = remove_ats_from_mol(at, sphere_around) # you are not your own neighbor
    for neighbor_at in sphere_around: # keep only atoms involved in bond
        bl_ref  = get_bl_from_map(neighbor_at[0], at[0])
        bl_calc = distance_between(neighbor_at, at)
        if not bl_ref * (1 - tolerance) < bl_calc < bl_ref * (1 + tolerance):
            sphere_around = remove_ats_from_mol(neighbor_at, sphere_around)

    return sphere_around, len(sphere_around) # neighbors and their number


def rem_ats_with_this_neighbor(cluster, fragment_to_update, at_type):
    cluster = np.array(cluster, dtype=object)
    fragment_updated = np.array(fragment_to_update, dtype=object)

    i = 0
    while i < len(fragment_updated):
        at = fragment_updated[i]
        cutoff = get_bl_from_map(at[0], at_type)
        neighbors, _ = get_neighbors(at, cluster, cutoff)
        if at_type in neighbors[:, 0].tolist():
            fragment_updated = remove_ats_from_mol(at, fragment_updated)
        else:
            i += 1
    return fragment_updated


def rem_ats_wrong_coord(xyz_data, custom_coord_map, ncheck, custom_cutoff=None):
    xyz_data = np.array(xyz_data, dtype=object)
    
    while ncheck > 0 and len(xyz_data) > 0: #ncheck do multiple rounds
        idx_to_remove = []

        for idx, at in enumerate(xyz_data):
            _, coord = get_neighbors(at, xyz_data, custom_cutoff)
            allowed = custom_coord_map.get(at[0], [])
            if coord not in allowed:
                idx_to_remove.append(idx)
        
        if not idx_to_remove:
            break # its fine, we stop the loop

        idx_to_keep = list(set(range(len(xyz_data))) - set(idx_to_remove))
        xyz_data = xyz_data[idx_to_keep] # quicker than calling remove_ats_from_mol
        ncheck -= 1
    return xyz_data


def remove_ats_from_mol(ats, mol, threshold=1e-3):
    if ats is None or len(ats) < 1:
        return mol
    
    ats = np.atleast_2d(ats) # handle single atoms given as list object
    mol_coords, _ = normalize_data_like(mol)
    ats_coords, _ = normalize_data_like(ats)

    coord_diffs = np.abs(mol_coords[:, np.newaxis, :] - ats_coords) # =0 for match atoms
    sum_diffs = np.sum(coord_diffs, axis=2)             # the sum of coord_diffs is zero
    matches_any = np.any(sum_diffs < threshold, axis=1) # get the removal mask

    return mol[~matches_any]


def is_at_in_mol(at, mol, threshold = 1e-3):
    at_coords = normalize_point_like(at)
    mol_coords, _ = normalize_data_like(mol)
    mask = np.sum(np.abs(mol_coords - at_coords), axis=1) < threshold
    # if atom with the same coordinates was detected, check its type
    return mol[mask][0, 0] == at[0] if np.any(mask) else False
    

def add_ats_to_mol(ats, mol):
    mol_arr = np.array(mol, dtype=object)
    if ats is None:
        return mol_arr

    ats_arr = np.atleast_2d(np.array(ats, dtype=object))
    if ats_arr.size == 0:
        return mol_arr

    return np.concatenate((mol_arr, ats_arr), axis=0)


def remove_duplicates(mol, threshold=1e-3):
    mol = np.array(mol, dtype=object)
    if len(mol) < 2:
        return mol
    unique_mol = [mol[0]] # initialization
    for any_at in mol[1:]:
        if not is_at_in_mol(any_at, np.array(unique_mol, dtype=object), threshold=threshold):
            unique_mol.append(any_at)
    # if not unique_mol:
    #     return np.empty((0, 4), dtype=object)
    return np.array(unique_mol, dtype=object)
    
    
def remove_None(data):
    rows_with_none = np.any(data == None, axis=1)
    return data[~rows_with_none]


def are_ats_in_mol(at, mol, threshold=1e-3):
    mol_arr = np.array(mol, dtype=object)

    # check if index(indices) are present in the molecule
    if isinstance(at, (int, np.integer)):
        return 0 <= at < len(mol_arr)
    if isinstance(at, (list, np.ndarray)) and len(at) > 0 and isinstance(at[0], (int, np.integer)):
        if len(np.array(at).shape) == 1:
            return [0 <= idx < len(mol_arr) for idx in at]
        else:
            return [0 <= idx < len(mol_arr) for idx in at.flatten()]

    # check if atom(s) with the same coordinates are present in the molecule
    at_arr = np.array(at, dtype=object)
    if len(at_arr.shape) == 1:
        return is_at_in_mol(at_arr, mol_arr, threshold=threshold)
    results = []
    results.extend(
        is_at_in_mol(single_at, mol_arr, threshold=threshold)
        for single_at in at_arr
    )
    return results


def replace_at_in_mol_by(at_or_idx, mol, at_or_el, threshold=1e-3):
    mol_arr = np.array(mol, dtype=object).copy()
    if isinstance(at_or_idx, (int, np.integer)): # replace by index
        if are_ats_in_mol(at_or_idx, mol): # are_ats and not is_at because index check
            rep_idx = at_or_idx
    else: # replace by position matching
        at_arr = np.array(at_or_idx, dtype=object)
        at_coords = normalize_point_like(at_arr)
        mol_coords, _ = normalize_data_like(mol_arr)
        coord_diffs = np.sum(np.abs(mol_coords - at_coords), axis=1)
        matches = np.where(coord_diffs < threshold)[0] # match coord. idx.

        if len(matches) == 0:
            raise ValueError("Atom to replace not found in molecule")
        if len(matches) > 1:
            raise ValueError("Multiple atoms match the given coordinates")

        rep_idx = matches[0]
        old_atom = mol_arr[rep_idx]
    # replacement atom
    at_or_el_arr = np.array(at_or_el, dtype=object)
    if isinstance(at_or_el, str): # only label provided, keep old coordinates
        new_atom = [at_or_el, old_atom[1], old_atom[2], old_atom[3]]
    elif len(at_or_el_arr) == 3: # only coordinates provided, keep element symbol
        new_atom = [old_atom[0], at_or_el_arr[0], at_or_el_arr[1], at_or_el_arr[2]]
    else: # full atom provided
        new_atom = at_or_el_arr.tolist()

    mol_arr[rep_idx] = new_atom
    return mol_arr
    

def get_closest_ats_in_mol(mol, ref_at, nclosest=1):
    mol_filtered = remove_ats_from_mol(ref_at, mol)
    if len(mol_filtered) == 0:
        return np.array([], dtype=object)
    distances = distance_between(ref_at, mol_filtered)
    indices = np.argsort(distances)[:min(int(nclosest), len(mol_filtered))]
    return mol_filtered[indices]


def get_furthest_ats_in_mol(mol, ref_at, nfurthest=1):
    mol_filtered = remove_ats_from_mol(ref_at, mol) # remove the ref_at form the cluster if present
    if len(mol_filtered) == 0:
        return np.array([], dtype=object)
    
    distances = distance_between(ref_at, mol_filtered)
    indices = np.argsort(distances)[-min(int(nfurthest), len(mol_filtered)):]
    return mol_filtered[indices]


def order_by_distance(mol, at_types=None):
    mol_labels, mol_coords = mol[:, 0], normalize_data_like(mol)[0]
    if at_types is None: # all atoms ordered by default
        distances = np.linalg.norm(mol_coords, axis=1)
        sorted_idx = np.argsort(distances)
        return mol[sorted_idx]
    # else sort only atoms types listed at_types, in same order
    mask = np.isin(mol_labels, at_types)
    in_at_types  = mol[mask]
    out_at_types = mol[~mask]
    
    distances = np.linalg.norm(in_at_types[:, 1:].astype(float), axis=1)
    sorted_idx = np.argsort(distances)
    in_at_types = in_at_types[sorted_idx]
    return np.concatenate((in_at_types, out_at_types), axis=0)


def add_at_along_axis(ref_at, e_z, add_at_type):
    bond_length = get_bl_from_map(add_at_type, ref_at[0])
    new_at = np.empty(4, dtype=object)
    new_at[0]  = add_at_type
    new_at[1:] = ref_at[1:] + bond_length * e_z
    return new_at
    

def add_at_on_bridge(ref_ats, e_para, add_at_type):
    ref_at_1, ref_at_2 = ref_ats[0], ref_ats[1]
    ref_pos_1, ref_pos_2 = normalize_point_like(ref_at_1), normalize_point_like(ref_at_2)
    
    bond_length = get_bl_from_map(add_at_type, ref_at_1[0])
    
    mid_bridge_pos = ref_pos_1 + (ref_pos_2 - ref_pos_1) / 2
    half_bridge_length = np.linalg.norm(ref_pos_2 - ref_pos_1) / 2
    vertical_length = np.sqrt(np.abs(bond_length**2 - half_bridge_length**2))

    new_at = np.empty(4, dtype=object)
    new_at[0] = add_at_type
    new_at[1:] = mid_bridge_pos + vertical_length * e_para
    return new_at


def add_at_in_frame(ref_at, e_z, e_y, add_at_type, theta):
    bond_length = get_bl_from_map(add_at_type, ref_at[0])
    new_at = np.empty(4, dtype=object)
    new_at[0]  = add_at_type
    new_at[1:] = ref_at[1:] + bond_length * (np.cos(theta) * e_z + np.sin(theta) * e_y)
    return new_at


def draw_rand_at_from_pool(rand_pool):
    idx = np.random.choice(len(rand_pool), size=1)[0]
    rand_pool_draw = rand_pool[idx]
    rand_pool_updated = remove_ats_from_mol(rand_pool_draw, rand_pool)
    return rand_pool_draw, rand_pool_updated


def align_zaxis(mol, zat1, zat2, zat1_origin=True):
    def _find(mol, zat1, zat2):
        mol_coords, _ = normalize_data_like(mol)
        zat1_idx = np.where(mol[:, 0] == zat1)[0]
        zat2_idx = np.where(mol[:, 0] == zat2)[0]
        nzat1, nzat2 = len(zat1_idx), len(zat2_idx)

        dist_mat = np.zeros((nzat1, nzat2))
        for i in range(nzat1): # compute distance matrix
            for j in range(nzat2):
                dist_mat[i, j] = cdist([mol_coords[zat1_idx[i]]],
                                       [mol_coords[zat2_idx[j]]])[0,  0]
                
        max_idx_flat = np.argmax(dist_mat) # zat1 and zat2 maximizing dist_mat
        max_idx_row, max_idx_col = np.unravel_index(max_idx_flat, dist_mat.shape)        
        
        zat1_pos = mol_coords[zat1_idx[max_idx_row]]
        zat2_pos = mol_coords[zat2_idx[max_idx_col]]
        return zat1_pos, zat2_pos

    # check types for the alignement process
    mol_types = mol[:, 0]
    avail_types = np.unique(mol_types)
    for ref_type in [zat1, zat2]:
        if ref_type not in avail_types: 
            raise ValueError(f"Ligand does not contain atom of type {ref_type}.")

    # get the positions of fragment intramolecular axis
    mol_coords, _ = normalize_data_like(mol)
    zat1_coords, zat2_coords = _find(mol, zat1, zat2)
    
    axis = zat2_coords - zat1_coords
    axis = axis / np.linalg.norm(axis)    
    target_axis = np.array([0., 0., 1.], dtype=float)
    rotation_matrix = quick_rot_mat(axis, target_axis) # to align axis and target axis

    centered_coords = mol_coords - zat1_coords         # zat1 at zero
    rotated_coords = np.dot(centered_coords, rotation_matrix.T)    
    
    if not zat1_origin:
        rotated_coords += zat1_coords

    return np.column_stack((mol_types, rotated_coords))


def align_zaxis_xyplane(mol, zat1, zat2, xat1, xat1_force_sel=None, zat1_origin=True):
    def _find(mol, zat1, zat2, xat1, xat1_force_sel):
        mol_coords, _ = normalize_data_like(mol) 
        zat1_idx = np.where(mol_types == zat1)[0]
        zat2_idx = np.where(mol_types == zat2)[0]
        xat1_idx = np.where(mol_types == xat1)[0]
    
        nzat1, nzat2 = len(zat1_idx), len(zat2_idx)
        dist_mat = np.zeros((nzat1, nzat2))
        for i in range(nzat1):
            for j in range(nzat2):
                dist_mat[i, j] = cdist([mol_coords[zat1_idx[i]]],
                                       [mol_coords[zat2_idx[j]]])[0,  0]
                
        max_idx_flat = np.argmax(dist_mat)
        max_idx_row, max_idx_col = np.unravel_index(max_idx_flat, dist_mat.shape)
        
        zat1_pos = mol_coords[zat1_idx[max_idx_row]]
        zat2_pos = mol_coords[zat2_idx[max_idx_col]]
    
        if xat1_force_sel:
            closest_idx = xat1_idx[xat1_force_sel] # force to take the xat1_force_sel-th at of type in mol
        else:
            zat1xat1_dist = cdist([zat1_pos], mol_coords[xat1_idx])[0]
            closest_idx = xat1_idx[np.argmin(zat1xat1_dist)]
        xat1_pos = mol_coords[closest_idx]
        return zat1_pos, zat2_pos, xat1_pos, closest_idx

    mol_types = mol[:, 0]
    avail_types = np.unique(mol[:, 0])
    for ref_type in [zat1, zat2, xat1]:
        if ref_type not in avail_types: 
            raise ValueError(f"Ligand does not contain atom of type {ref_type}.")

    mol_coords, _ = normalize_data_like(mol)
    zat1_coords, zat2_coords, _, closest_idx = _find(mol, zat1, zat2, xat1, xat1_force_sel)
    
    axis = zat2_coords - zat1_coords
    axis = axis / np.linalg.norm(axis)
    target_axis = np.array([0., 0., 1.], dtype=float)
    rotation_matrix = quick_rot_mat(axis, target_axis)
    
    centered_coords = mol_coords - zat1_coords
    rotated_coords = np.dot(centered_coords, rotation_matrix.T)
    
    xat1_rotated = rotated_coords[closest_idx]
    x, y, _ = xat1_rotated
    theta = -np.arctan2(y, x)
    z_rotation = arbitrary_rot_mat([0.0, 0.0, 1.0], theta)
        
    rotated_coords = np.dot(rotated_coords, z_rotation.T)

    if not zat1_origin:
        rotated_coords += zat1_coords
        
    return np.column_stack((mol_types, rotated_coords))
