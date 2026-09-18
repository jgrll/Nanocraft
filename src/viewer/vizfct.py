from .viz import _plot_py3Dmol, _basename
from ..nanocraft.ioxyz import load_xyz
from ..nanocraft.fcts import get_facets_from_family, _get_facet_id, generate_facets


def _check_facet(facet): # check [1,1,1] format
    return isinstance(facet, list) and len(facet) == 3 and all(isinstance(i, int) for i in facet)


def _check_family(family): # check'111' format
    return isinstance(family, str)


def _families_intersection(list_of_atom_indices, list_of_families):
    # need non empty intersection between atom indices and each of the families
    for family in list_of_families:
        family_facet_indices = [_get_facet_id(facet) for facet in get_facets_from_family(family)]
        if len(set(list_of_atom_indices) & set(family_facet_indices)) == 0:
            return False
    return True


def _families_union(list_of_atom_indices, list_of_families):
    # check if any of the atom indices belongs to any of the families
    list_of_family_indices = []
    for family in list_of_families:
        list_of_family_indices.extend([_get_facet_id(facet) for facet in get_facets_from_family(family)])
    return len(set(list_of_atom_indices) & set(list_of_family_indices)) > 0


def _facets_intersection(list_of_atom_indices, list_of_facets):
    # all of the facet indices must be in
    list_of_facet_indices = [_get_facet_id(facet) for facet in list_of_facets]
    return set(list_of_facet_indices) <= set(list_of_atom_indices)


def _facets_union(list_of_atom_indices, list_of_facets):
    # check if any of the facet indices of the atom is one of the accepted indices
    list_of_facet_indices = [_get_facet_id(facet) for facet in list_of_facets]
    return len(set(list_of_facet_indices) & set(list_of_atom_indices)) > 0


def visualize_facets(inputfiles, facet_or_family='all', intersection=False, dummy='U', **kwargs):# -> Any:
    
    if not isinstance(inputfiles, list):
        return visualize_facets([inputfiles], facet_or_family, intersection, dummy, **kwargs)

    if isinstance(facet_or_family, str):
        return visualize_facets(inputfiles, [facet_or_family], intersection, dummy, **kwargs)

    elif isinstance(facet_or_family, list) and len(facet_or_family) == 3 and all(isinstance(i, int) for i in facet_or_family):
        return visualize_facets(inputfiles, [facet_or_family], intersection, dummy, **kwargs)

    # now inputfiles is a list and facet_or_family must be a list of str / a list of 3 integers lists
    first_element = facet_or_family[0]
    if isinstance(first_element, str):
        all_valid = all(_check_family(family) for family in facet_or_family)
        _union, _intersection = _families_union, _families_intersection
    elif isinstance(first_element, list):
        all_valid = all(_check_facet(facet) for facet in facet_or_family)
        _union, _intersection = _facets_union, _facets_intersection
    else:
        raise ValueError("facet_or_family must be a list of strings (families) or a list of lists of 3 integers (facets)")
    if not all_valid:
        raise ValueError("Invalid facet_or_family format: all elements must be strings (families) or lists of 3 integers (facets)")

    _morgan = _intersection if intersection else _union

    list_of_xyz, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=True, center_COM=True)
        facets_info, _ = generate_facets(xyz_data)
        # loop on atom to replace it based on the chosen selection
        for i, facet_info in enumerate(facets_info):
            if facet_info[4] == "surf" and _morgan(facet_info[:4].tolist(), facet_or_family):
                xyz_data[i, 0] = dummy
        list_of_xyz.append(xyz_data)
        labels.append(_basename(inputfile))

    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)