import os
import numpy as np
from scipy.spatial.distance import cdist

from .log_cnfg import logger
from .utls import vec_diff_unit, compute_COM, to_unit_interval, settings_keys_check, handle_random_seed, get_random_angle, rotate_around_axis, normalize_data_like
from .ioxyz import load_xyz, get_filepath_for_xyz, get_formula_from_xyz, save_xyz, get_anchor_from_formula
from .cnfg import COORD_MAP, NC_TOP_DIR
from .molutls import distance_between, get_neighbors, order_by_distance, get_closest_ats_in_mol, remove_ats_from_mol, add_ats_to_mol, get_bl_from_map
from .srflyrs import surface_area_estimation, get_core_atoms_and_outer_layer, get_ligands_from_capping_layer, retrieve_labels_simplices, surface_tessellation


def _substitution_frame(cluster, fragment, anchor):
    origin_atom = order_by_distance(fragment[fragment[:, 0] == anchor])[0] # frame origin

    if fragment.shape[0] > 1:
        selected = get_closest_ats_in_mol(fragment, origin_atom, 1)[0]
    else:
        selected = get_closest_ats_in_mol(cluster, origin_atom, 2)[-1] # 2nd to unsure ex_p != ez for ontop atom
    ex_p = vec_diff_unit(origin_atom, selected) # temporary ex vector

    # ez vector for local frame
    typeSurfAtoms = list(COORD_MAP['surf_coord'].keys())
    cluster_work = remove_ats_from_mol(fragment, cluster)
    typeSurfMask = np.isin(cluster_work[:, 0], typeSurfAtoms)
    cluster_work = cluster_work[typeSurfMask]
    typeSurfNeigh, card = get_neighbors(origin_atom, cluster_work)
    if card == 0:
        reference_com = compute_COM(cluster_work)
    else:
        if card > 3:
            typeSurfNeigh = order_by_distance(typeSurfNeigh)[:3]
        reference_com = compute_COM(typeSurfNeigh)
    ez = vec_diff_unit(reference_com, origin_atom)
    global_com = compute_COM(cluster)
    inward = vec_diff_unit(origin_atom, global_com)
    if np.dot(ez, inward) > 0:
        ez = -ez

    ## additional check for ex_p
    # if abs(np.dot(ez, ex_p)) > 0.95:
    #     fallback = np.array([1.0, 0.0, 0.0])
    #     if abs(np.dot(ez, fallback)) > 0.95:
    #         fallback = np.array([0.0, 1.0, 0.0])
    #     ex_p = fallback

    ey = np.cross(ez, ex_p)
    ey /= np.linalg.norm(ey) # ez, ex_p not orthogonal, ey is not an unit vector
    ex = np.cross(ey, ez)
    return origin_atom[1:].astype(float), ex, ey, ez


def _substitute_ligands(xyz_cluster, start_fragment, end_fragment, start_anchor):
    start_fragment = np.atleast_2d(start_fragment)
    end_fragment   = np.atleast_2d(end_fragment)
    r0, ex, ey, ez = _substitution_frame(xyz_cluster, start_fragment, start_anchor)
    xyz_cluster = remove_ats_from_mol(start_fragment, xyz_cluster)
    R = np.stack([ex, ey, ez])
    end_fragment[:, 1:] = r0 + end_fragment[:, 1:].astype(float) @ R
    xyz_cluster = add_ats_to_mol(end_fragment, xyz_cluster)
    return xyz_cluster


def functionalize(settings, output_dir=None):
    valid_keys = {'ligand_file', 'nanocrystal_file', 'density_type', 'density_value',
                  'anchor_fragment', 'placement_method', 'random_angle', 'random_seed'}
    mandatory_keys = {'ligand_file', 'nanocrystal_file', 'density_type', 
                      'density_value','anchor_fragment', 'placement_method'}
    settings_keys_check(settings, valid_keys, mandatory_keys)

    ligand_file = settings['ligand_file']
    nanocrystal_file = settings['nanocrystal_file']
    density_value = settings['density_value']
    density_type = settings['density_type']
    anchor_fragment = settings['anchor_fragment']
    placement_method = settings['placement_method']
    random_seed  = settings['random_seed'] if 'random_seed' in settings else True
    random_angle = settings['random_angle'] if 'random_angle' in settings else True

    xyz_cluster = load_xyz(nanocrystal_file, sanity_check=True, center_COM=True)
    end_ligand  = load_xyz(ligand_file, sanity_check=True, center_COM=False)
    seed = handle_random_seed(random_seed)
    np.random.seed(seed)

    # get available ligands for substitution
    anchor = get_anchor_from_formula(anchor_fragment)
    if anchor not in np.unique(xyz_cluster[:, 0]): 
        raise ValueError("Invalid anchor fragment. Use this method to substitute existing ligands. Consider using 'functionalize_bare' method, or passivation first.")
    
    core, capping_layer = get_core_atoms_and_outer_layer(xyz_cluster, 'cap')  
    start_ligands = get_ligands_from_capping_layer(capping_layer, anchor_fragment, anchor)    
    nstart = len(start_ligands)
    nsubs = _compute_nsubs(density_type, density_value, nstart, core=core)

    # get the positions of the ligands to be substituted
    selected_indices = _select_sites(
        np.array([start_ligands[i][0] for i in range(len(start_ligands))]), 
        placement_method, nstart, nsubs)
    
    # substitute anchor fragment (old ligand) by the new ligands
    start_ligands_selected = [start_ligands[i] for i in selected_indices]
    for start_ligand in start_ligands_selected:
        end_ligand_updated = _apply_random_rotation(end_ligand, random_angle)
        xyz_cluster = _substitute_ligands(xyz_cluster, start_ligand, end_ligand_updated, anchor)
    
    # save to file  
    if output_dir is None:
        output_dir = os.path.join(NC_TOP_DIR, "functionalized")
    filepath = get_filepath_for_xyz(xyz_cluster, output_dir, f"{nsubs}L")
    header = f"{nsubs} ligands added ({get_formula_from_xyz(end_ligand)}), {placement_method} placement, random_angle={random_angle}, random_seed={random_seed}."
    save_xyz(xyz_cluster, filepath, header_message=header, remove_duplicates=False)


def functionalize_bare(settings, output_dir=None):
    # check si au moins un atom fait partie d'autre chose que le coeurs (in ligands)
    valid_keys = {'ligand_file', 'nanocrystal_file', 'density_type', 'density_value', 'anchor_type',
                  'anchor_site', 'placement_method', 'random_angle', 'random_seed', 'nloc_COM', 'alpha_tessel'}
    mandatory_keys = {'ligand_file', 'nanocrystal_file', 'density_type', 
                      'density_value','anchor_site', 'placement_method', 'anchor_type'}
    settings_keys_check(settings, valid_keys, mandatory_keys)

    ligand_file = settings['ligand_file']
    nanocrystal_file = settings['nanocrystal_file']
    density_value = settings['density_value']
    density_type = settings['density_type']
    anchor_site = settings['anchor_site']
    anchor_type = settings['anchor_type']
    placement_method = settings['placement_method']
    nloc_COM = settings['nloc_COM'] if 'nloc_COM' in settings else False
    random_seed  = settings['random_seed'] if 'random_seed' in settings else True
    random_angle = settings['random_angle'] if 'random_angle' in settings else True
    alpha = settings['alpha_tessel'] if 'alpha_tessel' in settings else 1.2

    xyz_cluster = load_xyz(nanocrystal_file, sanity_check=True, center_COM=True)
    end_ligand = load_xyz(ligand_file, sanity_check=True, center_COM=False)
    seed = handle_random_seed(random_seed)
    np.random.seed(seed)

    if not nloc_COM or not isinstance(nloc_COM, int):
        nloc_COM = int(0.15 * len(xyz_cluster))
    if isinstance(nloc_COM, int):
        nloc_COM = min(nloc_COM, len(xyz_cluster) - 1)

    _, capping_layer = get_core_atoms_and_outer_layer(xyz_cluster, 'cap')
    if len(capping_layer) > 0:
        logger.error("The cluster provided for 'functionalize_bare' is not bare. Consider using 'functionalize' method instead.")
        return # or raise error ?
    
    _, simplices, area, _ = surface_tessellation(xyz_cluster, method="alpha", alpha=alpha)
    labelled_simplices = retrieve_labels_simplices(xyz_cluster, simplices)
    anchor_points = _find_bare_sites(labelled_simplices, anchor_site, anchor_type)
    nmax = len(anchor_points)
    nsubs = _compute_nsubs(density_type, density_value, nmax, area=area)

    # draw the site to functionalize
    selected_indices = _select_sites(anchor_points, placement_method, nmax, nsubs)
    selected_points = anchor_points[selected_indices]

    # put a dummy, placeholder atom
    glb_com = compute_COM(xyz_cluster)
    offset = _vertical_offset(end_ligand, anchor_site, anchor_type)
    dummy_atoms = np.empty((nsubs, 4), dtype=object)
    for i, site in enumerate(selected_points):
        local_com = compute_COM(get_closest_ats_in_mol(xyz_cluster, site, nclosest=nloc_COM))
        local_ez = vec_diff_unit(local_com, site)
        if np.dot(site - glb_com, site - local_com) < 0. :
            local_ez *= -1.
        dummy_atoms[i, 0] = 'X' 
        dummy_atoms[i, 1:] = site + offset * local_ez
    xyz_cluster = add_ats_to_mol(dummy_atoms, xyz_cluster)

    # replace dummy atom by the proper ligand
    for dummy_atom in dummy_atoms:
        end_ligand_updated = _apply_random_rotation(end_ligand, random_angle)
        xyz_cluster = _substitute_ligands(xyz_cluster, dummy_atom, end_ligand_updated, str(dummy_atom[0]))

    # save to file  
    if output_dir is None:
        output_dir = os.path.join(NC_TOP_DIR, "functionalized_bare")
    filepath = get_filepath_for_xyz(xyz_cluster, output_dir, f"{nsubs}L")
    header = f"{nsubs} ligands added ({get_formula_from_xyz(end_ligand)}), {placement_method} placement, random_angle={random_angle}, random_seed={random_seed}."
    save_xyz(xyz_cluster, filepath, header_message=header, remove_duplicates=False)
    return


def _find_bare_sites(labelled_simplices, anchor, anchor_card):
    candidates, filtered = [], []
    for labelled_simplex in labelled_simplices:
        sel = labelled_simplex[:, 0] == anchor
        if np.sum(sel) >= anchor_card:
            candidates.append(np.atleast_2d(labelled_simplex[sel]))

    for site in candidates: # reduce order of simplices
        if site.shape[0] > anchor_card:
            if anchor_card == 1:
                for i in range(site.shape[0]):
                    filtered.append(np.atleast_2d(site[i]))
            elif anchor_card == 2:
                for i in range(site.shape[0]):
                    for j in range(i + 1, site.shape[0]):
                        filtered.append(np.atleast_2d([site[i], site[j]]))
        else:
            filtered.append(site)
    candidates, filtered = filtered, []

    if anchor_card <= 2: # unicity of binding sites
        sel_sites_com = np.array([np.mean(site[:, 1:], axis=0) for site in candidates], dtype=float)
        for idx, com in enumerate(sel_sites_com):
            if idx == 0:
                filtered.append(candidates[idx])
            else:
                dists = np.linalg.norm(sel_sites_com[:idx] - com, axis=1)
                if np.all(dists > 1e-3): # all visited com are different from the current one
                    filtered.append(candidates[idx])
        candidates, filtered = filtered, [] # update
        
    if anchor_card >= 2: # validity of binding sites
        ref_dist = get_bl_from_map(anchor, anchor)
        thr_lwr, upr = 0.5 * ref_dist, 1.7 * ref_dist
        for site in candidates:       
            if anchor_card == 2:
                dist = distance_between(site[0], site[1])
                if thr_lwr < dist < upr:
                    filtered.append(site)
            elif anchor_card == 3:
                dist01 = distance_between(site[0], site[1])
                dist02 = distance_between(site[0], site[2])
                dist12 = distance_between(site[1], site[2])
                if all(thr_lwr < d < upr for d in [dist01, dist02, dist12]):
                    filtered.append(site)
        candidates = filtered # update

    return np.array([np.mean(site[:, 1:], axis=0) for site in candidates], dtype=float)


def _select_sites(points, method, ntot, nsel):
    indices = np.arange(ntot)
    if len(points) < 1:
        return np.array([], dtype=int)
    points, _ = normalize_data_like(points)
    if method == "random":
        return np.random.choice(indices, nsel, replace=False)
    elif method == "distance":
        selected_indices = []
        selected_points = []
        first_idx = int(np.random.randint(ntot))
        selected_indices.append(first_idx)
        selected_points.append(points[first_idx])
        distances = np.full(ntot, np.inf)
        for _ in range(1, nsel):
            last_point = selected_points[-1].reshape(1, -1)
            new_distances = cdist(points, last_point).flatten()
            distances = np.minimum(distances, new_distances)
            next_idx = int(np.argmax(distances))
            selected_indices.append(next_idx)
            selected_points.append(points[next_idx])
            distances[next_idx] = 0
        return selected_indices
    else:
        raise ValueError("Unknown value for 'placement' argument.")
    

def _compute_nsubs(density_type, density_value, nmax, core=None, area=None):
    # Compute number of substitutions from density specification.
    if density_type == "absolute":
        try:   nsubs = int(density_value)
        except: raise TypeError("'density_value' should be of type int when 'density_type' is 'absolute'.")
    elif density_type == "percentage":
        try:   nsubs = round(to_unit_interval(density_value) * nmax)
        except: raise TypeError("'density_value' should be of type float when 'density_type' is 'percentage'.")
    elif density_type == "per_nm2":
        ref = area if area is not None else surface_area_estimation(core)
        try:   nsubs = round(density_value * ref)
        except: raise ValueError("Could not estimate the surface area of the nanocrystal.")
    else:
        raise ValueError("Invalid value for 'density_type'. Should be one of 'absolute', 'percentage', 'per_nm2'.")
    if nsubs > nmax:
        logger.warning(f"Requested number of ligands to be substituted ({nsubs}) >  maximum available ({nmax}). Defaulting to max. value.")
        nsubs = nmax
    return nsubs


def _vertical_offset(ligand, anchor_label, anchor_card):
    # get the distance between ligand first atom (closest to origin in internal coordinates)
    # and the fictious anchor site, based on the type of anchor and type of site (top, bridge,...)
    # cannot use get_closest_ats_in_mol since it ignores atom at COM
    sel = ligand[np.argmin(np.linalg.norm(ligand[:, 1:].astype(float), axis=1))]
    r_sel_anchor = get_bl_from_map(anchor_label, sel[0])
    r_anchor_anchor = get_bl_from_map(anchor_label, anchor_label)
    if anchor_card == 1:
        return r_sel_anchor
    elif anchor_card == 2: # equilateral triangle
        return np.sqrt(r_sel_anchor**2 - (0.5 * r_anchor_anchor)**2)
    elif anchor_card == 3: # regular triangular basis pyramid
        return np.sqrt(r_sel_anchor**2 - (r_anchor_anchor / np.sqrt(3))**2)
    else:
        raise ValueError("Unsupported value for 'anchor_card'. Should be 1, 2 or 3.")
    

def _apply_random_rotation(ligand, enabled):
    if enabled:
        return rotate_around_axis(ligand, [0., 0., 1.], get_random_angle())
    return ligand.copy()
