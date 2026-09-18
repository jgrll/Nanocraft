import os
import numpy as np
from .log_cnfg import logger
from scipy.spatial import ConvexHull
from alphashape import alphashape # pyright: ignore[reportMissingImports]
from .ioxyz import get_dict_from_formula, get_formula_from_xyz, load_xyz, get_filepath_for_xyz, save_xyz
from .utls import settings_keys_check, resolve_user_path, handle_random_seed, aggregate_dicts, to_unit_interval, compute_COM, normalize_data_like
from .cnfg import COORD_MAP, NC_TOP_DIR
from .molutls import *
######################################################
#   SRFLYRS MODULE
######################################################
#
#   METHODS INVOLVING A NEIGHBOR SEARCH (TESSEL BELOW)
#
def chain_detection(outer_layer, start_atom): 
    def _overdetect(atom, neighbors):
        avail_coord = max(COORD_MAP['ligs_coord'][atom[0]])-1 # one bond already satisfied
        if avail_coord < len(neighbors): # keep closests
            nrem = len(neighbors) - avail_coord
            removed = get_furthest_ats_in_mol(neighbors, atom, nfurthest=nrem)
            neighbors = remove_ats_from_mol(removed, neighbors)
        return neighbors
    chain, in_scope = np.empty((1,4), dtype=object), np.atleast_2d(start_atom.copy()) #initialization
    while len(in_scope) > 0:
        chain = add_ats_to_mol(in_scope, chain) # add detected atoms to the chain
        outer_layer = remove_ats_from_mol(in_scope, outer_layer) # remove them from avail. atoms
        next_scope = np.empty((1,4), dtype=object)
        for atom in in_scope:
            candidate, _ = get_neighbors(atom, outer_layer, tolerance=0.5)
            candidate    = _overdetect(atom, candidate)
            next_scope   = add_ats_to_mol(candidate, next_scope)
        next_scope = remove_duplicates(remove_None(next_scope)) # sanitize placeholder
        in_scope = next_scope
    return remove_None(chain)


def _not_outer_layer(mol):
    # check if the molecule as at least one atom of bulk type only - we don't have only the capping layer
    typesBulkOnly = set(COORD_MAP['bulk_coord'].keys()) - set(COORD_MAP['ligs_coord'].keys())
    typesToCheck  = np.unique(mol[:,0])
    return any(el in typesBulkOnly for el in typesToCheck)


def get_ligands_from_capping_layer(capping_layer, ligand_formula, anchor_label):
    # capping layer contains everythig apart from the NP core; get all fragment matching formula
    ligands, ligand_dict = [], get_dict_from_formula(ligand_formula)
    if _not_outer_layer(capping_layer): # we work with the capping layer
        raise ValueError("Issue in detectinng the capping layer for ligand substitution.")
    capping_layer_work = capping_layer.copy()
    while len(capping_layer_work[capping_layer_work[:, 0] == anchor_label]) > 0:
        start_atom = capping_layer_work[capping_layer_work[:, 0] == anchor_label][0]
        chain = chain_detection(capping_layer_work, start_atom)
        chain_dict = get_dict_from_formula(get_formula_from_xyz(chain))
        if chain_dict == ligand_dict:
            ligands.append(chain)
        capping_layer_work = remove_ats_from_mol(chain, capping_layer_work)
    return ligands


def get_species_in_outer_layer(cluster):
    # get capping fragments list (formulas) from the outer layer; need the full cluster
    core_atoms, capping_layer = get_core_atoms_and_outer_layer(cluster, 'cap')
    _, outer_core_atoms     = get_core_atoms_and_outer_layer(core_atoms, 'nc')
    # starting atom of each chain belong to a ligand and is binded to a outer bulk atom
    formula_list, chain_start = [], []
    for candidate in capping_layer:
        _, num = get_neighbors(candidate, outer_core_atoms, tolerance=0.15) # smaller tolerance since heavily binded to bulk
        if num > 0:
            chain_start.append(candidate)
    # detect the full chain for each starting position
    chain_start = np.array(chain_start, dtype=object)
    for atom in chain_start:
        chain = chain_detection(capping_layer, atom)
        formula_list.append(get_formula_from_xyz(chain))
        capping_layer = remove_ats_from_mol(chain, capping_layer)
    return {formula: formula_list.count(formula) for formula in formula_list}


def get_core_atoms_and_outer_layer(xyz_data, surf_type, updated_bulk_coord_map=None):
    bulk_coord_map = updated_bulk_coord_map or COORD_MAP['bulk_coord']
    ligs_coord_map = COORD_MAP['ligs_coord']
    typesOnlyInBulk   = list(bulk_coord_map.keys())
    typesOnlyInLig    = list(ligs_coord_map.keys() - bulk_coord_map.keys())
    typesInBulkAndLig = list(ligs_coord_map.keys() & bulk_coord_map.keys())
    
    if surf_type == 'cap': # surface=ligands; ligands atoms defined in ligs_coord_map.
        # initial attribution: the outer_layer gets all ligands-only atoms
        outer_layer = xyz_data[np.isin(xyz_data[:, 0], typesOnlyInLig)]
        core_layer = xyz_data[~np.isin(xyz_data[:, 0], typesOnlyInLig)]
        core_layer_work, outer_layer_work = core_layer.copy(), outer_layer.copy()
        # outer_layer is updated by adding atoms bounded to ligands-only atoms
        for at in outer_layer:  
            neighbors, _ = get_neighbors(at, core_layer)
            neighbors_keep = neighbors[np.isin(neighbors[:, 0], typesInBulkAndLig)]
            core_layer_work = remove_ats_from_mol(neighbors_keep, core_layer_work)
            outer_layer_work = add_ats_to_mol(neighbors_keep, outer_layer_work)
        core_layer, outer_layer = core_layer_work, outer_layer_work

    elif surf_type == 'nc':   # surface=outermost bulk atoms, **everything** else is bulk
        # first: isolate the ligands
        core_layer, outer_layer = get_core_atoms_and_outer_layer(xyz_data, 'cap', updated_bulk_coord_map)
        core_layer_work, outer_layer_work = core_layer.copy(), outer_layer.copy()
        # second: we remove all core atoms with lower coordination and put them in the outer_layer list
        for at in core_layer:
            neighbors, _ = get_neighbors(at, core_layer)
            coord_number = int(np.sum(np.isin(neighbors[:, 0], typesOnlyInBulk)))
            if coord_number in bulk_coord_map[at[0]]:
                core_layer_work = remove_ats_from_mol(at, core_layer_work)
                outer_layer_work = add_ats_to_mol(at, outer_layer_work)
        core_layer, outer_layer = outer_layer_work, core_layer_work # Swap on purpose

    elif surf_type == 'mix':  # surface='cap'+'nc'
        # first: get the ligands atoms
        core_layer, outer_layer = get_core_atoms_and_outer_layer(xyz_data, 'cap', updated_bulk_coord_map)
        if len(outer_layer) == 0: # if no ligand, then 'mix' equiv. 'nc'
            return get_core_atoms_and_outer_layer(xyz_data, 'nc', updated_bulk_coord_map)
        # second: get all bounded atom from the core to ligands atoms
        core_layer_work, outer_layer_work = core_layer.copy(), outer_layer.copy()
        for at in outer_layer:
            neighbors, _ = get_neighbors(at, core_layer)
            neighbors_keep = neighbors[np.isin(neighbors[:, 0], typesOnlyInBulk)]
            core_layer_work = remove_ats_from_mol(neighbors_keep, core_layer_work)
            outer_layer_work = add_ats_to_mol(neighbors_keep, outer_layer_work)
        core_layer, outer_layer = core_layer_work, outer_layer_work   

    else:
        logger.error("surf_type must be one of 'nc', 'cap', 'mix'.")
        raise ValueError("surf_type must be one of 'nc', 'cap', 'mix'.")

    return core_layer, outer_layer


######################################################
#
#   METHODS INVOLVING TESSELLATION ALGORITHM
#
def substitute(settings, output_dir=None):
    def _populations(mol, at1, at2):
        return len(mol[mol[:, 0] == at1]), len(mol[mol[:, 0] == at2])
    def _ratio(nat1, nat2):
        return round(nat1 / (nat1 + nat2), 2)

    valid_keys = {'nanocrystal_file', 'atom_to_replace', 'replacement_atom', 
                  'mixing_ratio', 'method', 'direction', 'random_seed'}
    mandatory_keys = {'nanocrystal_file', 'atom_to_replace', 'replacement_atom', 
                      'mixing_ratio', 'method'}
    settings_keys_check(settings, valid_keys, mandatory_keys)

    nanocrystal_file = settings['nanocrystal_file']
    type_old = settings['atom_to_replace']
    type_new = settings['replacement_atom']
    tar_ratio = to_unit_interval(settings['mixing_ratio'])
    method = settings['method']
    direction = settings['direction'] if 'direction' in settings else 'outside_in'
    random_seed = settings['random_seed'] if 'random_seed' in settings else True 

    resolve_user_path(nanocrystal_file, 'r')
    xyz_cluster = load_xyz(nanocrystal_file, center_COM=True, sanity_check=True)
    seed = handle_random_seed(random_seed)
    np.random.seed(seed)

    # value checks here
    accepted_values = ['inside_out', 'outside_in', 'random', 'layer']
    valid_params = all(val in accepted_values for val in [method, direction])
    valid_old_at_type = type_old in COORD_MAP['bulk_coord'].keys()
    if not valid_params or not valid_old_at_type:
        logger.error("direction/method/atom_to_replace parameters is incorrect.")
        raise ValueError("direction/method/atom_to_replace parameters is incorrect.")
    
    if str(type_new) == '':
        type_new = 'X' # placeholder for vacancy; removed at the end
        logger.warning("Replacement atom not specified: using 'X' as placeholder for vacancy; 'X' will be removed at the end of the process.")

    # we update the coordination map for using get_core_outer_layer
    if type_new not in COORD_MAP['bulk_coord'].keys(): 
        updated_coord_map = aggregate_dicts(COORD_MAP['bulk_coord'], {type_new: COORD_MAP['bulk_coord'][type_old]})
        logger.warning(f"Warning: '{type_new}' not found in coordination map. Using coordination values of '{type_old}'; you may want to include {type_new} in the coordination map for next manipulations.")
    else:
        updated_coord_map = COORD_MAP['bulk_coord']

    # initial checkup
    ini_ratio = _ratio(*_populations(xyz_cluster, type_old, type_new))
    now_ratio = ini_ratio
    if now_ratio <= tar_ratio: # leave function
        logger.info(f"No substitution needed, current ratio {now_ratio} <= target ratio {tar_ratio}.")
        return
    else:
        logger.info(f"initial fraction of {type_old} amongst {type_old}+{type_new}: {ini_ratio}.")
        logger.info(f"target : {tar_ratio}.")
    
    # substitution loop
    if method == 'random':
        pool = xyz_cluster[xyz_cluster[:, 0] == type_old]
        while now_ratio > tar_ratio and len(type_old) > 0:
            drawn, pool = draw_rand_at_from_pool(pool) # drawn atom, updated pool
            xyz_cluster = replace_at_in_mol_by(drawn, xyz_cluster, type_new)
            now_ratio   = _ratio(*_populations(xyz_cluster, type_old, type_new))
    
    elif method == 'layer': # get layers (of bulk) from ordered outermost to innermost (onion peeling)
        layers = []
        core_atoms, surf_atoms = get_core_atoms_and_outer_layer(xyz_cluster, 'nc', updated_coord_map) 
        layers.append(surf_atoms[surf_atoms[:, 0] == type_old])
        while len(core_atoms[core_atoms[:, 0] == type_old]) > 0:
            core_atoms, surf_atoms = get_core_atoms_and_outer_layer(core_atoms, 'nc', updated_coord_map)
            layers.append(surf_atoms[surf_atoms[:, 0] == type_old])
        layers = layers[::-1] if direction == "inside_out" else layers # direction choice
        for pool in layers: # each layer is a possible pool, stop when tar_ratio reached or all layers visited.
            while now_ratio > tar_ratio and len(pool) > 0: 
                drawn, pool = draw_rand_at_from_pool(pool)
                xyz_cluster = replace_at_in_mol_by(drawn, xyz_cluster, type_new)
                now_ratio = _ratio(*_populations(xyz_cluster, type_old, type_new))

    else:
        raise ValueError("This method does not exist yet / is not implemented.")
    
    if type_new == 'X': # remove placeholder
        xyz_cluster = remove_ats_from_mol(xyz_cluster[xyz_cluster[:, 0] == 'X'], xyz_cluster) 
    # save the new cluster
    end_ratio = _ratio(*_populations(xyz_cluster, type_old, type_new))
    logger.info(f"final fraction of {type_old} amongst {type_old}+{type_new}: {end_ratio}.")
    if not output_dir:
        output_dir = os.path.join(NC_TOP_DIR, "substituted")
    filepath = get_filepath_for_xyz(xyz_cluster, output_dir, suffix="substitution")
    header = f"Subst. '{type_old}'->'{type_new}', start/requested/end: {ini_ratio}/{tar_ratio}/{end_ratio}, {method} ({direction}), seed={seed}"
    save_xyz(xyz_cluster, filepath, header_message=header, remove_duplicates=False)


######################################################
#
#   METHODS INVOLVING TESSELLATION ALGORITHM
#
def surface_area_estimation(mol):
    _, _, surface, _ = surface_tessellation(mol, "hull")
    computed_surfaces = [surface]
    for alpha in np.arange(0.1, 2.0, 0.3):
        _, _, surface, _ = surface_tessellation(mol, "alpha", alpha)
        computed_surfaces.append(surface)
    return max(computed_surfaces)


def surface_tessellation(mol, method='hull', alpha=1.2):
    core_atoms, _ = get_core_atoms_and_outer_layer(mol, 'cap') # remove this ligands
    core_coords = normalize_data_like(core_atoms)[0] / 10.0    # stability of alpha shapes
    if method == "hull":
        hull = ConvexHull(core_coords)
        vertices_pos, simplices_pos = hull.points[hull.vertices], hull.points[hull.simplices]
        area, vol = hull.area, hull.volume
    elif method == "alpha":
        alpha_shape = alphashape(core_coords, alpha)
        vertices_pos = alpha_shape.vertices
        simplices_pos = alpha_shape.vertices[alpha_shape.faces]
        area, vol = alpha_shape.area, alpha_shape.volume
    else:
        raise ValueError("Tesselation method not yet implemented.")
    return vertices_pos * 10.0, simplices_pos * 10.0, area, vol


def max_radius_tessellation(simplices_or_mol):
    if simplices_or_mol.ndim == 2 and simplices_or_mol.shape[1] == 4:
        _, simplices, _, _ = surface_tessellation(simplices_or_mol, 'hull')
    else:
        simplices = simplices_or_mol
    rmax = np.linalg.norm(compute_COM(simplices[0])) # initialization
    for simplice in simplices[1:]:
        rtmp = np.linalg.norm(compute_COM(simplice)) # largest sphere that can fit
        rmax = max(rmax, rtmp)
    return rmax * 0.95 # avoid boundary issues


def retrieve_labels_simplices(mol, simplices, tolerance=0.01):
    labeled_simplices = np.empty((simplices.shape[0], 3, 4), dtype=object)
    for i, simplex in enumerate(simplices):
        for j, atom in enumerate(simplex):
            distances = distance_between(atom, mol)
            closest_in_mol = np.argmin(distances)
            if distances[closest_in_mol] < tolerance:
                label = mol[closest_in_mol, 0]
            else:
                label = 'X' # track error at the end
            labeled_simplices[i, j] = [label, float(atom[0]), float(atom[1]), float(atom[2])]
    if np.any(labeled_simplices[:, :, 0] == 'X'):
        logger.warning("Warning: some simplices vertices were not matched with any atom in the molecule.")
        raise ValueError("Some simplices vertices were not matched with any atom in the molecule.")
    return labeled_simplices
