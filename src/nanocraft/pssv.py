import numpy as np
from math import floor
import os

from .log_cnfg import logger
from .ioxyz import get_formula_from_xyz, load_xyz, get_dict_from_formula, save_xyz, get_filepath_for_xyz, get_anchor_from_formula
from .utls import handle_random_seed, vec_diff_unit, compute_COM, rodrigues_rot, settings_keys_check, adapt_list_to_length
from .cnfg import NC_TOP_DIR, OXBARE_MAP, COORD_MAP, DEFAULT_ANGLE
from .fcts import get_facets_from_family, is_facet_defined, get_sub_facet_info, load_facets, get_ats_from_facet
from .molutls import *
from .srflyrs import get_core_atoms_and_outer_layer, get_species_in_outer_layer, get_ligands_from_capping_layer


def get_cluster_ox(cluster, oxidation_estim=None):
    if oxidation_estim is not None:
        if oxidation_estim == 'surface_mix': # outer bulk from OXBARE_MAP, ligs values from coordination.json 
            core, ligs = get_core_atoms_and_outer_layer(cluster, 'cap')
            _, surf = get_core_atoms_and_outer_layer(core, 'nc')
            ligs_contrib = sum(COORD_MAP['ligs_coord'].get(at[0], 0)[0]*_get_sign(at[0]) for at in ligs)
            surf_contrib = sum(OXBARE_MAP.get(at[0], 0) for at in surf)
            return ligs_contrib + surf_contrib
        elif oxidation_estim == 'surface_ox': # ligs + outer bulk values from OXBARE_MAP
            _, selected = get_core_atoms_and_outer_layer(cluster, 'mix')
            return sum(OXBARE_MAP.get(at[0], 0) for at in selected)
        elif oxidation_estim == 'all_ox': # all atoms values from OXBARE_MAP (core included)
            return sum(OXBARE_MAP.get(at[0], 0) for at in cluster)
        else:
            logger.error(f"oxidation_estim '{oxidation_estim}' is not valid.")
            raise ValueError(f"oxidation_estim '{oxidation_estim}' is not valid.")
    else:
        logger.warning("No oxidation_estim provided. Oxydation state will be evaluated as 0.0 (neutral) by default.")
        logger.info(f"FYI: 'surface_mix' -> {get_cluster_ox(cluster, 'surface_mix')}")
        logger.info(f"FYI: 'surface_ox' -> {get_cluster_ox(cluster, 'surface_ox')}")
        logger.info(f"FYI: 'all_ox' -> {get_cluster_ox(cluster, 'all_ox')}")
        return 0.0


def get_mol_ox(molecule, oxidation_estim=None):
    # oxidation degree of isolated molecule (not a crystal)
    if oxidation_estim is not None:
        if oxidation_estim == 'surface_mix': # ligs values from coordination.json 
            return sum(COORD_MAP['ligs_coord'].get(at[0], 0)[0]*_get_sign(at[0]) for at in molecule)
        elif oxidation_estim in ['surface_ox', 'all_ox']: # ligs values from OXBARE_MAP
            return sum(OXBARE_MAP.get(at[0], 0) for at in molecule)
        else:
            logger.error(f"oxidation_estim '{oxidation_estim}' is not valid.")
            raise ValueError(f"oxidation_estim '{oxidation_estim}' is not valid.")
    else:
        logger.warning("No oxidation_estim provided. Oxydation state will be evaluated as 0.0 (neutral) by default.")
        logger.info(f"FYI: 'surface_mix' -> {get_mol_ox(molecule, 'surface_mix')}")
        logger.info(f"FYI: 'surface_ox' / 'all_ox' -> {get_mol_ox(molecule, 'surface_ox')}")
        return 0.0


def _get_sign(label):
    # get the sign for coordination based oxidation; copies the sign in atomic.dat 
    ox = OXBARE_MAP.get(label, 0)
    return 1 if ox > 0 else -1 if ox < 0 else 0


def _pseudorandom_removal(capped_cluster, rem_formula, nrem): 
    cap_anchor = get_anchor_from_formula(rem_formula)
    core_atoms, capping_atoms = get_core_atoms_and_outer_layer(capped_cluster, 'cap')
    capping_groups = get_ligands_from_capping_layer(capping_atoms, rem_formula, cap_anchor)
    
    # we need to know where to store the ligands based on the coordination of the closest bulk atom
    all_coords_surf     = list(COORD_MAP['surf_coord'].values())
    all_coords_collapse = [i for sub in all_coords_surf for i in sub] # single list
    all_coords_collapse = list(set(all_coords_collapse))              # remove duplicates
    bulk_coord_incr_map = {coord: i for i, coord in enumerate(sorted(all_coords_collapse, reverse=True))}
    
    # thanks to bulk_coord_incr_map, we can sort the capping groups based on the coordination of the bulk atom they are binded to
    capping_groups_pools = [[] for _ in range(len(bulk_coord_incr_map))]
    for capping_group in capping_groups:
        reference_atom = capping_group[0]
        bulk_atom_closest = get_closest_ats_in_mol(core_atoms, reference_atom, nclosest=1)[0]
        _, coord = get_neighbors(bulk_atom_closest, core_atoms)
        if coord not in COORD_MAP['surf_coord'][bulk_atom_closest[0]]:
            logger.warning(f"Bulk atom {bulk_atom_closest} has unexpected coordination {coord}. Skipped for pseudorandom removal.")
        else:
            capping_groups_pools[bulk_coord_incr_map.get(coord)].append(capping_group)

    # now we shuffle all of the pools and add them back to a single list from which we just take the from nrem_true elements
    nrem_avail = sum(len(pool) for pool in capping_groups_pools)
    nrem_true  = min(nrem, nrem_avail)
    capping_groups_shuffled_by_pool = []
    for pool in capping_groups_pools:
        shuffle_pool = pool.copy()
        np.random.shuffle(shuffle_pool)
        capping_groups_shuffled_by_pool.extend(shuffle_pool)
    for group in capping_groups_shuffled_by_pool[:nrem_true]:
        capped_cluster = remove_ats_from_mol(group, capped_cluster)
    return capped_cluster


def _random_removal(capped_cluster, rem_formula, nrem): 
    cap_anchor = get_anchor_from_formula(rem_formula)
    _, capping_atoms = get_core_atoms_and_outer_layer(capped_cluster, 'cap')
    capping_groups   = get_ligands_from_capping_layer(capping_atoms, rem_formula, cap_anchor)
    nrem_avail = len(capping_groups)
    nrem_true  = min(nrem, nrem_avail)
    indices = np.arange(nrem_avail)
    selected_indices = np.random.choice(indices, nrem_true, replace=False)
    for i in selected_indices:
        selected_group = capping_groups[i]
        capped_cluster = remove_ats_from_mol(selected_group, capped_cluster)
    return capped_cluster


def get_bridge_sites_def(cluster, atoms_of_facet):
    if len(atoms_of_facet) < 2:
        return []
    np.random.shuffle(atoms_of_facet)
    bridge_sites, dist_history = [], []
    atom_init_1 = atoms_of_facet[0]
    atom_init_2 = get_closest_ats_in_mol(atoms_of_facet, atom_init_1)[0]
    dist_history.append(distance_between(atom_init_1, atom_init_2))
    while len(atoms_of_facet) > 1: # need at least two atoms to define a bridge
        ref_atom  = atoms_of_facet[0]
        cls_atoms = get_closest_ats_in_mol(atoms_of_facet, ref_atom, nclosest=2)    
        if len(cls_atoms) < 2:
            cls_atom = cls_atoms[0]
        else:
            nb_ref, _  = get_neighbors(ref_atom, cluster)
            nb_cls2, _ = get_neighbors(cls_atoms[1], cluster)
            if any(are_ats_in_mol(nb_cls2, nb_ref)): # cls_atoms[1] (furthest) shares a neighbor with ref_atom
                cls_atom = cls_atoms[1]
            else:
                cls_atom = cls_atoms[0] # resort to closest
        # we check if the distance is not abnormally high compared to previous ones (init. supposed correct)
        dist_ref_cls = distance_between(ref_atom, cls_atom)
        if dist_ref_cls > np.mean(dist_history) * 1.35: # > 1.41 would allow diagonal sites
            atoms_of_facet = remove_ats_from_mol(ref_atom, atoms_of_facet)
        else:
            bridge_sites.append([ref_atom, cls_atom])
            dist_history.append(dist_ref_cls)
            atoms_of_facet = remove_ats_from_mol([ref_atom, cls_atom], atoms_of_facet)
    return np.array(bridge_sites)


def _sites_removal(cluster, oxidation_estim, neutral_reach):
    if neutral_reach == 'full':
        logger.info("neutral_reach set to 'full': no passivation group will be removed.")
        return cluster
    elif neutral_reach == 'random':
        removal_method = _random_removal
    elif neutral_reach == 'pseudorandom':
        removal_method = _pseudorandom_removal
    else:
        raise ValueError("neutral_reach must be either 'full', 'random' or 'pseudorandom'.")
    
    start_ox = get_cluster_ox(cluster, oxidation_estim)
    logger.info(f"Oxydation degree (before {neutral_reach} sites removal): {start_ox}")
    if start_ox == 0:
        logger.info("Cluster is already neutral, no passivation group was removed.")
        return cluster

    # evaluate the oxidation of all available groups
    group_count = get_species_in_outer_layer(cluster)
    logger.info(f"Available passivation groups (with counts): {group_count}")
    group_ox = group_count.copy()
    for formula in group_count.keys():
        formula_dict = get_dict_from_formula(formula)
        dummy_system = [k for k, v in formula_dict.items() for _ in range(v)]
        dummy_system = np.array((dummy_system), dtype=object).reshape(-1, 1) # xyz array, only labels
        group_ox[formula] = get_mol_ox(dummy_system, oxidation_estim)

    if start_ox > 0: # keep only groups whose charge is of the same sign as start_ox
        filtered_group_ox = {group: ox for group, ox in group_ox.items() if ox > 0}
    else:
        filtered_group_ox = {group: ox for group, ox in group_ox.items() if ox < 0}

    filtered_group_count = {
        group: count 
        for group, count in group_count.items() 
        if group in filtered_group_ox.keys()
    }

    if not filtered_group_count: # no passiv. groups with right charge to remove
        logger.warning("No oxydation groups of adequate sign available for removal to reach neutrality.")
        return cluster
    else: # sorted groups by decreasing absolute oxydation and by decreasing abundance when ox. is identical
        group_sorted = sorted(filtered_group_ox.items(),                                
                        key=lambda item: (-abs(item[1]), -filtered_group_count[item[0]])) 
        group_sorted = [group for group, _ in group_sorted]
        logger.info(f"Relevant passivation groups (oxydation degree): {filtered_group_ox}")

    nrem_dict = {} # get the number of groups to remove for each type
    for candidate in group_sorted: 
        this_ox = filtered_group_ox[candidate]
        this_avail = filtered_group_count[candidate]
        this_needed = floor(start_ox / this_ox)
        this_nrem = min(this_needed, this_avail)
        nrem_dict[candidate] = this_nrem
        start_ox -= this_nrem * this_ox # update
        if start_ox <= 0: 
            break

    logger.info(f"Following groups will be removed (nrem times): {nrem_dict}")
    logger.info(f"Final oxydation degree (reachable): {start_ox}")
    for group, nrem in nrem_dict.items():
        cluster = removal_method(cluster, group, nrem=nrem)
    return cluster


def _sites_selection(cluster, facet_info, binding_site, binding_type, facet_constrain, coord_constrain):
    # Successive filtering on atoms to get availbale sites for passivation
    valid_binding_types, valid_facet_constrains = ['ontop', 'bridge'], ['all', '100', '111', '110']
    if binding_type not in valid_binding_types:
        logger.error(f"Error: binding_type '{binding_type}' is not valid. Choose from {valid_binding_types}.")
        raise ValueError(f"Error: binding_type '{binding_type}' is not valid. Choose from {valid_binding_types}.")
    if facet_constrain not in valid_facet_constrains:
        logger.error(f"Error: facet_constrain '{facet_constrain}' is not valid. Choose from {valid_facet_constrains}.")
        raise ValueError(f"Error: facet_constrain '{facet_constrain}' is not valid. Choose from {valid_facet_constrains}.")

    bulk_atoms, capping_atoms = get_core_atoms_and_outer_layer(cluster, 'cap')   # allow successive partial passivation
    _, candidates = get_core_atoms_and_outer_layer(bulk_atoms, 'nc') # initial (full) pool for this step

    # 1. filter on atom type
    atomic_type_mask = candidates[:, 0] == binding_site
    candidates = candidates[atomic_type_mask] 

    # 2. filter already passivated sites from the pool (bond with ligands)
    for atom in capping_atoms:                                                   
        to_remove, _ = get_neighbors(atom, candidates)
        candidates = remove_ats_from_mol(to_remove, candidates)

    # 3. filter on coordination
    if coord_constrain == 'all' or isinstance(coord_constrain, int):
        if coord_constrain != 'all':
            to_remove = []
            for remaining in candidates:
                _, coord = get_neighbors(remaining, bulk_atoms)
                if coord != coord_constrain:
                    to_remove.append(remaining)
            candidates = remove_ats_from_mol(to_remove, candidates)
    else:
        raise ValueError("coord_constrain must either be 'all' or an integer.")

    # 4. filter on facet family
    ## - facet_constrain == 'all' and binding_type == 'ontop' 
    if facet_constrain == 'all' and binding_type == 'ontop':
        return candidates 
    ## Remaining cases: 
    ## - facet_constrain != 'all' and binding_type == 'ontop' 
    ## - facet_constrain != 'all' and binding_type == 'bridge'
    ## - facet_constrain == 'all' and binding_type == 'bridge'
    candidates_facet_info = get_sub_facet_info(cluster, facet_info, candidates)
    facets_candidates = []
    facets_selected = (get_facets_from_family(facet_constrain) if facet_constrain != 'all'
                        else get_facets_from_family('100') 
                        + get_facets_from_family('111')
                        + get_facets_from_family('110'))
    facets_filtered = [facet if is_facet_defined(candidates_facet_info, facet) 
                       else None 
                       for facet in facets_selected]
    facets_filtered = list(filter(None, facets_filtered)) # keep only the defined facets
    for facet in facets_filtered:
        selected = get_ats_from_facet(candidates, candidates_facet_info, facet)
        if len(selected) > 0:
            facets_candidates.append(selected) # selected is always (N, 4)
    
    if binding_type == 'ontop':
        if not facets_candidates:
            return np.array([], dtype=object)
        facets_candidates = np.vstack(facets_candidates)
        facets_candidates = remove_duplicates(facets_candidates)
        return facets_candidates
    
    bridge_facets = [] # remain bridge cases
    for candidates_this_facet in facets_candidates:
        bridge_this_facet = get_bridge_sites_def(bulk_atoms, candidates_this_facet)
        bridge_facets.extend(bridge_this_facet)
    bridge_facets = _bridge_unicity(bridge_facets)
    return np.array(bridge_facets)


def _bridge_unicity(bridges):
    # detect bridge duplicates by comparing COM's
    filtered = []
    bridge_com = np.array([np.mean(bridge[:, 1:].astype(float), axis=0) for bridge in bridges])
    for idx, com in enumerate(bridge_com):
        if idx == 0:
            filtered.append(bridges[idx])
        else:
            dists = np.linalg.norm(bridge_com[:idx] - com, axis=1)
            if np.all(dists > 1e-3): # all visited com are different from the current one
                filtered.append(bridges[idx])
    return filtered


def _sites_passivation(cluster, sites, fragment):
    sites = np.array(sites, dtype=object)
    if sites.ndim <= 1:   # empty list
        return cluster
    elif sites.ndim == 2: # list of atoms for ontop
        add_method = add_group_on_top
    elif sites.ndim == 3: # list of pairs of atoms for bridge
        add_method = add_group_on_bridge
    else:
        raise ValueError("Invalid sites array shape.")
    fragment_dict   = get_dict_from_formula(fragment)
    fragment_labels = [label for label, count in fragment_dict.items() for _ in range(count)]
    if not 0 < len(fragment_labels) <= 3:
        raise ValueError(f"Fragment '{fragment}' has too many atoms.")
    for site in sites:
        cluster = add_method(cluster, site, fragment_labels)
    return cluster


def _get_eperp_pool(e_para, e_perp_start, nperp):
    thetas  = [2*k*np.pi/nperp for k in range(nperp)]
    return [rodrigues_rot(e_perp_start, e_para, theta) for theta in thetas]


def _get_test_positions_pool(ref_at, ez, eys, add_at_type, theta):
    pool = np.empty((len(eys), 4), dtype=object)
    for idx in range(pool.shape[0]):
        pool[idx] = add_at_in_frame(ref_at, ez, eys[idx], add_at_type, theta)
    return pool


def _select_minimal_hindrance(pool, cluster, reference_atom):
    pool_label = pool[0, 0]
    already_in_cluster = cluster[cluster[:, 0] == pool_label]
    if already_in_cluster.shape[0] < 1:
        return get_furthest_ats_in_mol(pool, ['X', 0.0, 0.0, 0.0])
    closest_existing = get_closest_ats_in_mol(already_in_cluster, reference_atom, nclosest=1)[0]
    return get_furthest_ats_in_mol(pool, closest_existing, nfurthest=1)[0]


def add_group_on_top(cluster, reference_atom, labels, nperp=12):    
    neighbors, _ = get_neighbors(reference_atom, cluster)
    if len(neighbors) > 1:
        neighbors_COM = compute_COM(neighbors)
    else:
        neighbors_COM = compute_COM(cluster)
    e_para = vec_diff_unit(neighbors_COM, reference_atom) # vertical axis in local frame
    first_atom = add_at_along_axis(reference_atom, e_para, labels[0]) # always along axis
    cluster  = add_ats_to_mol(first_atom, cluster)

    if len(labels) == 1: #single atom added, we stop here
        return cluster
    if len(labels) == 2: # diatomic case
        theta = np.pi - np.deg2rad(get_ba_from_map(reference_atom[0], labels[0], labels[1]))
        remaining_atoms_angles = [(labels[1], theta)] # di/tri-atomic is derived from list len from now on
    else: # triatomic case
        theta_below = get_ba_from_map(reference_atom[0], labels[0], labels[1])
        theta_opening = get_ba_from_map(labels[0], labels[1], labels[2])
        if abs(theta_opening - DEFAULT_ANGLE) > 1e-3: # theta_opening was given in config file
            theta = np.deg2rad(theta_opening) / 2.0   # takes priority over theta_below
        else:
            theta = np.pi - np.deg2rad(theta_below)
        remaining_atoms_angles = [(labels[1], theta), (labels[2], theta)]
 
    # possible positions for second atom in both remaining cases
    e_perp_start = vec_diff_unit(neighbors_COM, neighbors[0])
    e_perp_pool  = _get_eperp_pool(e_para, e_perp_start, nperp)
    for label, angle in remaining_atoms_angles: 
        pool = _get_test_positions_pool(first_atom, e_para, e_perp_pool, label, angle)
        selected_atom = _select_minimal_hindrance(pool, cluster, first_atom)
        cluster = add_ats_to_mol(selected_atom, cluster)
    return cluster


def add_group_on_bridge(cluster, reference_atoms, labels, nperp=12):
    neighbors, _ = get_neighbors(reference_atoms[0], cluster)
    if neighbors.shape[0] > 1:
        neighbors_COM = compute_COM(neighbors)
    else:
        neighbors_COM = compute_COM(cluster) # take the global COM
    e_z  = vec_diff_unit(neighbors_COM, reference_atoms[0])
    first_atom = add_at_on_bridge(reference_atoms, e_z, labels[0])
    cluster = add_ats_to_mol(first_atom, cluster)
    
    if len(labels) == 1:
        return cluster
    elif len(labels) == 2:
        second_atom = add_at_along_axis(first_atom, e_z, labels[1])
        cluster = add_ats_to_mol(second_atom, cluster)
        return cluster
    # triatomic case
    theta = np.deg2rad(get_ba_from_map(labels[0], labels[1], labels[2])) / 2.0    
    e_inter = vec_diff_unit(reference_atoms[1], reference_atoms[0])  # local ex
    e_perp  = np.cross(e_z, e_inter)                                 # local ey
    e_perp /= np.linalg.norm(e_perp)
    e_perps = _get_eperp_pool(e_z, e_perp, nperp)

    for label in [labels[1], labels[2]]:
        pool = _get_test_positions_pool(first_atom, e_z, e_perps, label, theta)
        selected_atom = _select_minimal_hindrance(pool, cluster, first_atom)
        cluster = add_ats_to_mol(selected_atom, cluster)
    return cluster


def _process_scheme(passiv_scheme):
    processed = []
    if not isinstance(passiv_scheme, list):
      return _process_scheme([passiv_scheme])
    for item in passiv_scheme:
        if (isinstance(item, tuple) and len(item) == 2):
            processed.append(item)
        else:
            logger.error(f"Error: Invalid item '{item}' in passiv_scheme. Please update.")
            raise TypeError(f"Error: Invalid item '{item}' in passiv_scheme. Please update.")
    return processed


def passivate(settings, output_dir=None):
    valid_keys = {'nanocrystal_file', 'passiv_scheme', 'binding_site', 'oxidation_method',
                  'facet_constrain', 'coord_constrain', 'neutral_reach', 'random_seed'}
    mandatory_keys = {'nanocrystal_file', 'passiv_scheme', 'binding_site'}
    settings_keys_check(settings, valid_keys, mandatory_keys)

    nanocrystal_file = settings['nanocrystal_file']
    passiv_scheme   = settings['passiv_scheme']
    binding_site    = settings['binding_site']                                                  
    facet_constrain = settings['facet_constrain'] if 'facet_constrain' in settings else None 
    coord_constrain = settings['coord_constrain'] if 'coord_constrain' in settings else None 
    oxidation_estim = settings['oxidation_method'] if 'oxidation_method' in settings else 'all_ox'
    neutral_reach   = settings['neutral_reach'] if 'neutral_reach' in settings else 'full'     
    random_seed     = settings['random_seed'] if 'random_seed' in settings else True

    xyz_cluster = load_xyz(nanocrystal_file, center_COM=True, sanity_check=True)
    cluster_facets, _ = load_facets(nanocrystal_file)
    seed = handle_random_seed(random_seed)
    np.random.seed(seed)

    passiv_scheme = _process_scheme(passiv_scheme) # always a list, of size at least 1
    nsteps = len(passiv_scheme)       # len of each item in set below must much nsteps
    facet_constrain = ['all'] * nsteps if facet_constrain is None else facet_constrain
    coord_constrain = ['all'] * nsteps if coord_constrain is None else coord_constrain
    sets_to_process = [binding_site, facet_constrain, coord_constrain] 
    for i, set_to_process in enumerate(sets_to_process):
        sets_to_process[i] = adapt_list_to_length(set_to_process, nsteps)
    binding_site, facet_constrain, coord_constrain = sets_to_process # update

    logger.info(f"Starting oxydation degree (before passivation): {get_cluster_ox(xyz_cluster, oxidation_estim)}")
    for i in range(nsteps):
        anchor, fragment = passiv_scheme[i] # unpack tuple
        avail_sites = _sites_selection(xyz_cluster, cluster_facets, 
                                      anchor, binding_site[i], 
                                      facet_constrain[i], coord_constrain[i])
        xyz_cluster = _sites_passivation(xyz_cluster, avail_sites, fragment)
    xyz_cluster     = _sites_removal(xyz_cluster, oxidation_estim, neutral_reach) #  passiv groups not needed (auto. detected)

    if output_dir is None:
        output_dir = os.path.join(NC_TOP_DIR, "passivated")
    filepath = get_filepath_for_xyz(xyz_cluster, output_dir)
    header = f"Passivation using {[step[-1] for step in passiv_scheme]} on {[step[0] for step in passiv_scheme]}, constrains: {list(zip(facet_constrain, coord_constrain, binding_site))}. Ox.: {neutral_reach}, {oxidation_estim}, seed={random_seed}."
    save_xyz(xyz_cluster, filepath, header_message=header, remove_duplicates=False)
