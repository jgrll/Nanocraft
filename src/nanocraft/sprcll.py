import numpy as np
import math
from pprint import pprint
from .log_cnfg import logger
from .cnfg import COORD_MAP
from .ioxyz import load_xyz, check_coord_map
from .utls import resolve_user_path, compute_COM
from .srflyrs import max_radius_tessellation
from .molutls import get_closest_ats_in_mol, remove_ats_from_mol, order_by_distance
from pymatgen.io.cif import CifParser # pyright: ignore[reportMissingImports]


def load_supercell(inputfile, supercell_size=None):
    if inputfile.lower().endswith('.xyz'):
        supercell = load_xyz(inputfile, center_COM=True, sanity_check=True)
    elif inputfile.lower().endswith('.cif'):
        supercell = _load_supercell_from_cif(inputfile, supercell_size)
    else:
        raise ValueError("Unsupported file format. Please provide an .xyz or .cif file.")
    max_radius = max_radius_tessellation(supercell)
    print(f"""
    Supercell loaded successfully from {inputfile}.
    FYI, this supercell can fit a sphere of radius: {round(max_radius, 3)} Å.
    Choose you parameter accordingly.
    """)
    nsnap = min(20, len(supercell))
    snapshot_atoms = get_closest_ats_in_mol(supercell, ['X', 0.0, 0.0, 0.0], nsnap)
    print(f"Snapshot of {nsnap} atoms closest to the supercell's COM:")
    pprint(snapshot_atoms)
    logger.info("Supercell loaded/generated successfully.")
    return order_by_distance(supercell) # faster validity checks later on when cutting


def _check_supercell_size(supercell_size):
    condition_1 = isinstance(supercell_size, list)
    condition_2 = len(supercell_size) == 3
    condition_3 = all(isinstance(i, int) and i > 0 for i in supercell_size)
    if not all([condition_1, condition_2, condition_3]):
        logger.error("Invalid supercell_size: must be a list of three positive integers.")
        raise ValueError("For CIF files, supercell_size must be a list of three positive integers.")


def _duplicate_cell_sites(conventional_cell, n1, n2, n3):
    nat = len(conventional_cell)
    ntrans = n1 * n2 * n3
    total_atoms = nat * ntrans
    u, v, w = np.meshgrid(range(n1), range(n2), range(n3), indexing='ij')
    translations = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=1)
    supercell_array = np.empty((total_atoms, 4), dtype=object)
    label  = conventional_cell[:, 0]
    coords = conventional_cell[:, 1:].astype(float)
    for trans_idx, (h, k, l) in enumerate(translations):
        start_idx = trans_idx * nat
        end_idx = (trans_idx + 1) * nat
        translated_coords = coords.copy()
        translated_coords[:, 0] = (translated_coords[:, 0] + h) / n1
        translated_coords[:, 1] = (translated_coords[:, 1] + k) / n2
        translated_coords[:, 2] = (translated_coords[:, 2] + l) / n3
        supercell_array[start_idx:end_idx, 0] = label
        supercell_array[start_idx:end_idx, 1:] = translated_coords
    return supercell_array


def _update_lattice_vector(lattice_params, n1, n2, n3):
    return {'a': lattice_params['a'] * n1,
            'b': lattice_params['b'] * n2,
            'c': lattice_params['c'] * n3}


def _fract2cart(fractional_cell, lattice_params, cell_angles):
    a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
    alpha, beta, gamma = cell_angles['alpha'], cell_angles['beta'], cell_angles['gamma']
    volume = a * b * c * math.sqrt(
        1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 +
        2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    )
    transformation_matrix = np.array([
        [a, b * math.cos(gamma), c * math.cos(beta)],
        [0, b * math.sin(gamma), c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)],
        [0, 0, volume / (a * b * math.sin(gamma))]
    ])
    labels = fractional_cell[:, 0]
    fractional_coords = fractional_cell[:, 1:].astype(float)
    cartesian_coords = fractional_coords @ transformation_matrix.T
    return np.hstack((labels[:, np.newaxis], cartesian_coords))


def _load_supercell_from_cif(cif_file, supercell_size):
    resolve_user_path(cif_file, 'r')      # file exists
    _check_supercell_size(supercell_size) # supercell size is valid
    parser = CifParser(cif_file)
    structure = parser.parse_structures(primitive=False)[0]
    lattice = structure.lattice

    lattice_params = {'a': lattice.a, 
                      'b': lattice.b, 
                      'c': lattice.c}
    cell_angles = {'alpha': math.radians(lattice.alpha), 
                   'beta':  math.radians(lattice.beta), 
                   'gamma': math.radians(lattice.gamma)}

    conventional_cell = []
    for site in structure:
        label  = site.species_string
        coords = site.frac_coords
        conventional_cell.append([label, coords[0], coords[1], coords[2]])
    conventional_cell = np.array(conventional_cell, dtype=object) # fractional coordinates
    if not check_coord_map(np.unique(conventional_cell[:, 0]).tolist()):
        print(label)
        raise ValueError("XYZ file contains atom types missing in coordination.json.")

    supercell_atoms = _duplicate_cell_sites(conventional_cell, *supercell_size) # output still in fractional coordinates
    supercell_lattice = _update_lattice_vector(lattice_params, *supercell_size) # the lattice needs to be updated
    supercell_atoms_cart = _fract2cart(supercell_atoms, supercell_lattice, cell_angles) # convert to cartesian
    supercell_atoms_cart[:, 1:] -= compute_COM(supercell_atoms_cart)    

    return supercell_atoms_cart


def apply_supercell_transform(xyz_supercell, central_site=None):
    def _center_on_atom(xyz_supercell, label):
        mask = xyz_supercell[:, 0] == label
        xyz_supercell[:, 1:] -= order_by_distance(xyz_supercell[mask])[0, 1:]
        return xyz_supercell
    
    def _center_on_point(xyz_supercell, point):
        xyz_supercell[:, 1:] -= point
        return xyz_supercell
    
    def _center_on_subsystem(xyz_supercell, labels):
        if len(labels) < 2: # handle list with only one type
            return _center_on_atom(xyz_supercell, labels[0])
        
        pool = xyz_supercell.copy()
        subsystem = np.empty((len(labels), 4), dtype=object)
        
        mask_init = pool[:, 0] == labels[0]
        init_COM = ['X', 0.0, 0.0, 0.0]
        subsystem[0] = get_closest_ats_in_mol(pool[mask_init], init_COM)[0]
        pool = remove_ats_from_mol(subsystem[0], pool)
        for i in range(1, len(subsystem)):
            mask = pool[:, 0] == labels[i]
            tmp_COM = ['X'] + list(compute_COM(subsystem[:i]))
            subsystem[i] = get_closest_ats_in_mol(pool[mask], tmp_COM)[0]
            pool = remove_ats_from_mol(subsystem[i], pool)
        xyz_supercell[:, 1:] -= compute_COM(subsystem)
        return xyz_supercell
  
    typeBulkAtoms = set(COORD_MAP['bulk_coord'].keys())
    if central_site in [None, 'None', 'none']:
        logger.debug("Supercell was centered on 'none' (global COM).")
        return xyz_supercell
    
    elif central_site == 'random':
        logger.debug("Supercell was centered on random point in a cube of side 1 nm")
        return _center_on_point(xyz_supercell, np.random.uniform(-5, 5, 3))

    elif isinstance(central_site, str) and central_site in typeBulkAtoms:
        logger.debug(f"Supercell was centered on most central {central_site}.")
        return _center_on_atom(xyz_supercell, central_site)

    elif (isinstance(central_site, (list, tuple))
          and len(central_site) == 3 
          and all(isinstance(c, (int, float)) for c in central_site)):
        logger.debug(f"Supercell was centered at {central_site} (relative to previous global COM).")
        return _center_on_point(xyz_supercell, np.array(central_site))

    elif (isinstance(central_site, (list, tuple)) 
          and (all(isinstance(a, str) 
                   and a in typeBulkAtoms for a in central_site))):
        logger.debug(f"Supercell was centered at the COM of most central {central_site}.")
        return _center_on_subsystem(xyz_supercell, central_site)

    raise ValueError(f"Invalid central_site: {central_site}.")
