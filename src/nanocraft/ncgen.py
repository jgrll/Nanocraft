import numpy as np
import os 

from .log_cnfg import logger
from .utls import aggregate_dicts, settings_keys_check
from .cnfg import COORD_MAP, NC_TOP_DIR, LABEL_SET_ORDERED
from .ioxyz import get_filepath_for_xyz, save_xyz
from .fcts import generate_facets
from .molutls import get_neighbors, rem_ats_wrong_coord
from .sprcll import apply_supercell_transform

#############################################
#   NCGEN MODULE
#   Add you own filter here !
#   To do make sure to:
#   1. Define a function that cuts the desired shape from the bulk data (like cut_shape_from_bulk)
#   2. Add the function to the accepted_mask_shapes dictionary in generate, with arguments and flags.
#   3. If the shape has facets, set save_facets to True and implement the facet generation function (like generate_faces).
#   Note: Regarding 3., you might need to update facet recognition as well (if i,j,k's are not in {+1, 0, -1}).
#############################################

def cut_tetrahedron_from_bulk(data, Rcut):
    # cut a perfect cube from bulk data
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    trunc_1 = x + y + z <= Rcut
    trunc_2 = -x - y + z <= Rcut
    trunc_3 = -x + y - z <= Rcut
    trunc_4 = x - y - z <= Rcut    
    combined_mask = trunc_1 & trunc_2 & trunc_3 & trunc_4
    return data[combined_mask]


def cut_octahedron_from_bulk(data, Rcut):
    # cut a perfect cube from bulk data
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    trunc_1 =  np.abs(x + y + z)  <= Rcut
    trunc_2 =  np.abs(-x - y + z) <= Rcut
    trunc_3 =  np.abs(-x + y - z) <= Rcut
    trunc_4 =  np.abs(x - y - z)  <= Rcut    
    combined_mask = trunc_1 & trunc_2 & trunc_3 & trunc_4
    return data[combined_mask]


def cut_sphere_from_bulk(data, Rcut):
    # Rcut acts as the sphere radius
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    mask = (x ** 2 + y ** 2 + z ** 2) ** 0.5 <= Rcut
    return data[mask]


def cut_cylinder_from_bulk(data, Rcut, F):
    # length is along z axis, radius in xy plane
    length, radius = F, Rcut
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    mask_length = np.abs(z) <= length / 2
    mask_radius = (x ** 2 + y ** 2) ** 0.5 <= radius
    combined_mask = mask_length & mask_radius
    return data[combined_mask]


def cut_cube_from_bulk(data, Rcut):
    # Rcut acts as the half side length of the cube
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    mask_x, mask_y, mask_z = np.abs(x) <= Rcut, np.abs(y) <= Rcut, np.abs(z) <= Rcut
    combined_mask = mask_x & mask_y & mask_z
    return data[combined_mask]


def cut_truncated_octahedron_from_bulk(data, R100, F):
    # Cut a truncated octahedron / truncated cube from bulk data.
    R111 = F * R100 * np.sqrt(2)
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    trunc_x, trunc_y, trunc_z = np.abs(x) <= R100, np.abs(y) <= R100, np.abs(z) <= R100
    trunc_1 =  np.abs(x + y + z)  <= R111
    trunc_2 =  np.abs(-x + y + z) <= R111
    trunc_3 =  np.abs(x - y + z)  <= R111
    trunc_4 =  np.abs(x + y - z)  <= R111
    trunc_5 =  np.abs(x - y - z)  <= R111
    trunc_6 =  np.abs(-x - y + z) <= R111
    trunc_7 =  np.abs(-x + y - z) <= R111
    trunc_8 =  np.abs(-x - y - z) <= R111
    combined_mask = trunc_x & trunc_y & trunc_z & trunc_1 & trunc_2 & trunc_3 & trunc_4 & trunc_5 & trunc_6 & trunc_7 & trunc_8
    return data[combined_mask]


def cut_stellated_from_bulk(data, Rcut):
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    # atoms must belong to either one of two equivalent tetrahedra
    trunc_1 = x + y + z <= Rcut
    trunc_2 = -x - y + z <= Rcut
    trunc_3 = -x + y - z <= Rcut
    trunc_4 = x - y - z <= Rcut

    tetra_1 = trunc_1 & trunc_2 & trunc_3 & trunc_4
    trunc_5 = -x + y + z <= Rcut
    trunc_6 = x - y + z <= Rcut
    trunc_7 = x + y - z <= Rcut
    trunc_8 = -x - y - z <= Rcut
    tetra_2 = trunc_5 & trunc_6 & trunc_7 & trunc_8

    combined_mask = tetra_1 | tetra_2
    return data[combined_mask]


def _check_if_config_valid(xyz_data, custom_coord_map):
    for at in xyz_data:
        _, coordination = get_neighbors(at, xyz_data)
        if coordination not in custom_coord_map[at[0]]:
            return False # config not valid, atom must be removed
    return True


def _check_acceptance(cluster, acceptance_criterion, nmin=12):
    agg_coord_map = aggregate_dicts(COORD_MAP['bulk_coord'], COORD_MAP['surf_coord'])
    match acceptance_criterion:
        case 'none':
            will_be_saved = True
        case 'tolerant':
            cluster = rem_ats_wrong_coord(cluster, agg_coord_map, ncheck=3)
            will_be_saved = True
        case 'strict':
            will_be_saved = _check_if_config_valid(cluster, agg_coord_map)
        case 'auto':
            will_be_saved = _check_if_config_valid(cluster, agg_coord_map) # first check for init
            while not will_be_saved:
                cluster = rem_ats_wrong_coord(cluster, agg_coord_map, ncheck=1)
                will_be_saved = _check_if_config_valid(cluster, agg_coord_map)
        case _:
            logger.error("Error: Invalid acceptance_criterion keyword.")
            raise ValueError("Invalid acceptance_criterion keyword.")
        
    if len(cluster) <= nmin: # minimal number of atoms
        will_be_saved = False
    return cluster, will_be_saved


def _subdir_naming(mask_shape, central_site):
    # simple check only (extensive check in sprcll:apply_supercell_transform)
    if isinstance(central_site, list):
        if isinstance(central_site[0], str):
            return f"{mask_shape}_{''.join(central_site)}"
        elif isinstance(central_site[0], (int, float)):
            return f"{mask_shape}_{'_'.join([str(x) for x in central_site])}"
    return f"{mask_shape}_{str(central_site)}"
    

def _header_naming(mask_shape, rcut, fratio=None):
    header = f"{mask_shape},Rc={round(rcut/10, 2)}"
    if fratio is not None:
        header += f",f={round(fratio, 2)}"
    return header


# Main function to generate a nanocrystal from bulk data, with a given shape and acceptance criterion.
#
def generate(settings, supercell, output_dir=None, remove_duplicates=True):
    valid_keys = {'mask_shape', 'rcut', 'fratio', 'acceptance_criterion', 'central_site'}
    mandatory_keys = valid_keys
    settings_keys_check(settings, mandatory_keys, mandatory_keys)

    rcut = settings['rcut'] * 10.                           # nm to angstrom
    fratio = settings['fratio']                             # octahedron truncation ratio (R111/R100)
    mask_shape = settings['mask_shape']                     # initial cut shape
    central_site = settings['central_site']                 # value verified in sprcll:apply_supercell_transform
    acceptance_criterion = settings['acceptance_criterion'] # value verified in check_acceptance

    accepted_mask_shapes = {
        "sphere": {
            "args": [rcut],
            "cut_shape_from_bulk": cut_sphere_from_bulk,
            "save_facets": False
        }, 
        "cylinder": {
            "args": [rcut, fratio],
            "cut_shape_from_bulk": cut_cylinder_from_bulk,
            "save_facets": False
        },
        "cube": {
            "args": [rcut],
            "cut_shape_from_bulk": cut_cube_from_bulk,
            "save_facets": True
        },
        "truncated_octahedron": {
            "args": [rcut, fratio],
            "cut_shape_from_bulk": cut_truncated_octahedron_from_bulk,
            "save_facets": True
        },
        "octahedron": {
            "args": [rcut],
            "cut_shape_from_bulk": cut_octahedron_from_bulk,
            "save_facets": True
        },
        "tetrahedron": {
            "args": [rcut],
            "cut_shape_from_bulk": cut_tetrahedron_from_bulk,
            "save_facets": True
        },
        "stellated": {
            "args": [rcut],
            "cut_shape_from_bulk": cut_stellated_from_bulk,
            "save_facets": True
        },
    }

    if mask_shape not in accepted_mask_shapes.keys():
        logger.error("Error: Invalid polyhedral_mask keyword.")
        raise ValueError("Invalid polyhedral_mask keyword.")

    supercell = apply_supercell_transform(supercell, central_site)
    cluster = accepted_mask_shapes[mask_shape]['cut_shape_from_bulk'](supercell, *accepted_mask_shapes[mask_shape]['args'])
    cluster, save_cluster = _check_acceptance(cluster, acceptance_criterion)

    if save_cluster:
        # sort by atom type based on the map
        key = np.array([LABEL_SET_ORDERED.get(label, float("inf")) for label in cluster[:, 0]], dtype=float)
        ordered = np.argsort(key)
        cluster = cluster[ordered]
        # parent directory
        subdir = _subdir_naming(mask_shape, central_site)
        if output_dir:
            output_dir = os.path.join(output_dir, subdir)
        else:
            output_dir = os.path.join(NC_TOP_DIR, "nanocrystals", subdir)
        # header naming
        header = _header_naming(mask_shape, rcut, fratio)
        filepath = get_filepath_for_xyz(cluster, output_dir)
        save_xyz(cluster, filepath, header_message=header, remove_duplicates=remove_duplicates)
        # generate facets if wanted
        if accepted_mask_shapes[mask_shape]['save_facets']:
            generate_facets(cluster, os.path.join(output_dir, "facets"))
