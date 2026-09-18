import numpy as np
import os
from .log_cnfg import logger
from .utls import resolve_user_path, spheric2cart, vec_diff_unit, to_unit_interval, settings_keys_check,resolve_user_path, handle_random_seed, all_same_type, adapt_list_to_length
from .cnfg import AMU_MAP, NC_TOP_DIR
from .ioxyz import load_xyz, get_filepath_for_xyz, save_xyz
from .molutls import get_neighbors, replace_at_in_mol_by, get_ats_ids
from .srflyrs import get_core_atoms_and_outer_layer


def _get_bias(el1, el2):
    heavy_el_ref = 'O'
    light_el_ref = 'H'
    kappa = 0.2
    reduced_mass = (AMU_MAP[heavy_el_ref] - AMU_MAP[light_el_ref]) / (AMU_MAP[el1] + AMU_MAP[el2])
    lambda_12 = min([kappa, kappa * reduced_mass])
    if AMU_MAP[el1] < AMU_MAP[el2]:
        lambda_12 *= -1
    return lambda_12


def _get_biased_deltas(delta_tot, el1, el2, bias=True): 
    if bias and abs(AMU_MAP[el1] - AMU_MAP[el2]) > 4.0: # avoid inversion
        bias_1, bias_2 = _get_bias(el1, el2), _get_bias(el2, el1) # opposite sign for bias_1 and bias_2
    else:
        bias_1, bias_2 = 0., 0.
    delta_1 = ( AMU_MAP[el2] / (AMU_MAP[el1] + AMU_MAP[el2]) + bias_1 ) * delta_tot
    delta_2 = ( AMU_MAP[el1] / (AMU_MAP[el1] + AMU_MAP[el2]) + bias_2 ) * delta_tot
    return delta_1, delta_2


def _shuffle_atom(at, rmax):
    rho, theta, phi = np.random.uniform(low=0.0, high=1.0, size=3)
    rho *= rmax
    theta *= np.pi
    phi *= 2*np.pi
    x, y, z = spheric2cart(rho, theta, phi)
    shuffled = at.copy()
    shuffled[1] = at[1] + float(x)
    shuffled[2] = at[2] + float(y)
    shuffled[3] = at[3] + float(z)
    return np.array(shuffled, dtype=object)


def _shuffle_bond(bond, rmax):
    start_at, end_at = bond[0], bond[1]
    axis = vec_diff_unit(start_at, end_at)
    delta_tot = np.random.uniform(low=-1.0, high=1.0, size=1) * rmax
    delta_1, delta_2 = _get_biased_deltas(delta_tot, start_at[0], end_at[0], bias=True)

    start_at_updated, end_at_updated = start_at.copy(), end_at.copy()
    start_at_updated[1:] = start_at[1:] - delta_1 * axis
    end_at_updated[1:] = end_at[1:] + delta_2 * axis
    return np.array([start_at_updated, end_at_updated], dtype=object)


def _bond_unicity(bond_list):
    # detect bonds duplicates by comparing COM's
    filtered = []
    bond_com = np.array([np.mean(bond[:, 1:].astype(float), axis=0) for bond in bond_list])
    for idx, com in enumerate(bond_com):
        if idx == 0:
            filtered.append(bond_list[idx])
        else:
            dists = np.linalg.norm(bond_com[:idx] - com, axis=1)
            if np.all(dists > 1e-3):
                filtered.append(bond_list[idx])
    return filtered


def _get_atoms_or_bonds_to_shuffle(xyz_cluster, layer, atom_or_bond, percent_shuffle):  
    percent_shuffle = to_unit_interval(percent_shuffle)
    if layer == 'bulk': # get all canditates
        lyr_pool, _ = get_core_atoms_and_outer_layer(xyz_cluster, 'cap')
    elif layer in ['nc', 'cap', 'mix']:
        _, lyr_pool = get_core_atoms_and_outer_layer(xyz_cluster, layer)
    elif layer == 'all':
        lyr_pool = xyz_cluster
    else:
        logger.error("Error: layer must be 'bulk', 'nc', 'cap', 'mix' or 'all'. Please update.")
        raise ValueError("Error: layer must be 'bulk', 'nc', 'cap', 'mix' or 'all'. Please update.")

    if isinstance(atom_or_bond, str): # if atoms to shuffle, atom_or_bond is the element symbol
        pool = lyr_pool[lyr_pool[:,0] == atom_or_bond]
        npool = len(pool)
        nshuffle = round(npool * percent_shuffle)
        indices_pool = np.random.choice(npool, nshuffle, replace=False)
        indices_global = get_ats_ids(pool[indices_pool], xyz_cluster)
        return indices_global # more reliable than positions that can change after shuffling (especially for bonds)
    
    atom_or_bond = list(atom_or_bond) # tuple to list
    pool = lyr_pool[np.isin(lyr_pool[:, 0], atom_or_bond)] # bonds to shuffle
    pool_1 = pool[pool[:,  0] == atom_or_bond[0]]
    pool_2 = pool[pool[:,  0] == atom_or_bond[1]]
    if len(pool_1) * len(pool_2) < 1:  # less than one atom of each type
        return np.array([], dtype=object)
    bnds_pool = []
    for atom in pool_1:
        candidates, _ = get_neighbors(atom, pool_2)
        bnds_pool.extend([np.array([atom, candidate]) for candidate in candidates])
    if atom_or_bond[0] == atom_or_bond[1]:
        bnds_pool = _bond_unicity(bnds_pool)
    npool = len(bnds_pool)
    nshuffle = round(npool * percent_shuffle)
    indices_pool = np.random.choice(npool, nshuffle, replace=False)
    bnds_pool_filtered = [bnds_pool[id] for id in indices_pool]
    bnds_indices_global = [get_ats_ids(bnd, xyz_cluster) for bnd in bnds_pool_filtered]
    return bnds_indices_global


def _process_moves(atoms_or_bonds_set):
    processed = []
    if not isinstance(atoms_or_bonds_set, list): 
        return _process_moves([atoms_or_bonds_set]) 
    for item in atoms_or_bonds_set:
        if isinstance(item, str) or (isinstance(item, tuple) and len(item) == 2):
            processed.append(item)
        else:
            message = f"Error: Invalid item '{item}' in atoms_bonds_set. Please update."
            logger.error(message)
            raise TypeError(message)
    return processed

# main function
#
def shuffle(settings, output_dir=None):
    valid_keys = {'xyz_file', 'layer', 'atoms_bonds_set', 'percent_shuffle', 'max_disp', 'random_seed'}
    mandatory_keys = {'xyz_file', 'layer', 'atoms_bonds_set', 'percent_shuffle', 'max_disp'}
    settings_keys_check(settings, valid_keys, mandatory_keys)

    xyz_file = settings['xyz_file']
    layer_set = settings['layer']
    atoms_bonds_set = settings['atoms_bonds_set']
    percent_shuffle_set = settings['percent_shuffle']
    max_disp_set = settings['max_disp']
    random_seed = settings['random_seed'] if 'random_seed' in settings else True

    resolve_user_path(xyz_file, 'r')
    xyz_cluster = load_xyz(xyz_file, center_COM=True, sanity_check=True)
    seed = handle_random_seed(random_seed)
    np.random.seed(seed)

    # verify types and number of argument for each parameter
    atoms_bonds_set = _process_moves(atoms_bonds_set) # atoms_bonds_set is now a list in all cases        
    sets_to_process = [layer_set, percent_shuffle_set, max_disp_set] # len of each item must be consistent
    types_of_values_in_sets = [str, float, float]
    for i, (set_to_process, type_of_values_in_set) in enumerate(zip(sets_to_process, types_of_values_in_sets)):
        if not all_same_type(set_to_process, type_of_values_in_set):
            logger.error("Error: All sets must contain values of the same type.")
            raise TypeError("Error: All sets must contain values of the same type.")
        sets_to_process[i] = adapt_list_to_length(set_to_process, len(atoms_bonds_set))
    layer_set, percent_shuffle_set, max_disp_set = sets_to_process # update sets

    # shuffle section
    # more robust to work with indices of atoms to shuffle rather than positions that can change after shuffling multiple times; 
    sets_to_process = [layer_set, atoms_bonds_set, percent_shuffle_set, max_disp_set]
    for layer, atom_or_bond, percentage, rmax in zip(*sets_to_process):
        selected = _get_atoms_or_bonds_to_shuffle(xyz_cluster, layer, atom_or_bond, percentage)
        if np.array([selected]).ndim == 2:   # list of atoms indices
            for sel in selected:
                shuffled_at = _shuffle_atom(xyz_cluster[sel], rmax)
                xyz_cluster = replace_at_in_mol_by(sel, xyz_cluster, shuffled_at)
        elif np.array([selected]).ndim == 3: # list of bonds (atoms indices)
            for sel in selected:
                bond = np.array([xyz_cluster[sel[0]], xyz_cluster[sel[1]]], dtype=object)
                shuffled_bond = _shuffle_bond(bond, rmax)
                xyz_cluster = replace_at_in_mol_by(sel[0], xyz_cluster, shuffled_bond[0])
                xyz_cluster = replace_at_in_mol_by(sel[1], xyz_cluster, shuffled_bond[1])
        else:
            logger.info(f"Nothing was shuffled ({layer}, {atom_or_bond}, {percentage}, {rmax}).")
        
    # saving instructions
    if output_dir is None:
        output_dir = os.path.join(NC_TOP_DIR, "shuffle")
    filepath = get_filepath_for_xyz(xyz_cluster, output_dir, suffix="shuffle")
    header = f"Shuffle[at,(bond)]: {atoms_bonds_set}, layer: {layer_set}, %: {percent_shuffle_set}, rmax: {max_disp_set}, seed: {seed}"
    save_xyz(xyz_cluster, filepath, header_message=header, remove_duplicates=False)
