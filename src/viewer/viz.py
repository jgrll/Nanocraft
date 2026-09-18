import py3Dmol
import numpy as np
import sys
import os
from ..nanocraft.ioxyz import load_xyz
from ..nanocraft.srflyrs import get_core_atoms_and_outer_layer


def _array2block(atoms):
    n_atoms = len(atoms)
    lines = [str(n_atoms), "Generated from numpy array"]
    for row in atoms:
        symbol = str(row[0])
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        lines.append(f"{symbol:<4s} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    return "\n".join(lines)  # corrected: was "\\n"


def _basename(path):
    return os.path.basename(path)


def _plot_py3Dmol(list_of_xyz, labels=None, ncols=5, style="default", color_scheme="Jmol", 
                  width=1600, height=200, background="white", spin=False):
    
    xyzs = [_array2block(xyz) for xyz in list_of_xyz]
    n_struct = len(xyzs)
    nrows = (n_struct + ncols - 1) // ncols

    if labels is None:
        labels = [f"structure {i+1}" for i in range(n_struct)]
    print(labels)

    viewer = py3Dmol.view(
        viewergrid=(nrows, ncols),
        width=width,
        height=height * nrows,
        linked=False
        )
    
    for i, (xyz, label) in enumerate(zip(xyzs, labels)):
        row, col = divmod(i, ncols)
        viewer.addModel(xyz, "xyz", viewer=(row, col))

        if style == "sphere": 
            viewer.setStyle({"sphere": {"colorscheme": color_scheme, "scale": 0.4}}, viewer=(row, col))
        elif style == "stick":
            viewer.setStyle({"stick": {"colorscheme": color_scheme}}, viewer=(row, col))
        elif style == "line":
            viewer.setStyle({"line": {"colorscheme": color_scheme}}, viewer=(row, col))
        elif style == "cross":
            viewer.setStyle({"cross": {"colorscheme": color_scheme, "lineWidth": 5}}, iewer=(row, col))
        else:
            viewer.setStyle({"stick": {"colorscheme": color_scheme, "radius": 0.12},
                             "sphere": {"colorscheme": color_scheme, "scale": 0.3}}, viewer=(row, col))

        viewer.setBackgroundColor(background)
        viewer.zoomTo(viewer=(row, col))

        viewer.addLabel(label, {
            "font": "sans-serif", "fontSize": 16, "fontColor": "black" if background == "white" else "white",
            "backgroundColor": "white" if background == "white" else "black", "backgroundOpacity": 0.6,
            "showBackground": True, "inFront": True, "position": {"x": 0, "y": 0, "z": 0},}, viewer=(row, col)
            )

        if spin:
            viewer.spin(True)

    viewer.show()
    # return viewer


def visualize_xyz(inputfiles, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_xyz([inputfiles], **kwargs)
    list_of_xyz = [load_xyz(inputfile, sanity_check=False, center_COM=True) for inputfile in inputfiles]
    labels = [_basename(inputfile) for inputfile in inputfiles]
    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)


def visualize_nc_core(inputfiles, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_nc_core([inputfiles], **kwargs)
    list_of_xyz, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=True, center_COM=True)
        xyz_bulk, _ = get_core_atoms_and_outer_layer(xyz_data, surf_type='cap')
        list_of_xyz.append(xyz_bulk)
        labels.append(_basename(inputfile))
    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)


def visualize_nc_surf(inputfiles, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_nc_surf([inputfiles], **kwargs)
    list_of_xyz, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=True, center_COM=True)
        _, xyz_surf = get_core_atoms_and_outer_layer(xyz_data, surf_type='nc')
        list_of_xyz.append(xyz_surf)
        labels.append(_basename(inputfile))
    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)


def visualize_nc_ligands(inputfiles, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_nc_ligands([inputfiles], **kwargs)
    list_of_xyz, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=True, center_COM=True)
        _, xyz_ligs = get_core_atoms_and_outer_layer(xyz_data, surf_type='cap')
        list_of_xyz.append(xyz_ligs)
        labels.append(_basename(inputfile))
    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)


def visualize_nc_all(inputfiles, **kwargs):
    # sourcery skip: merge-list-appends-into-extend
    if not isinstance(inputfiles, list):
        return visualize_nc_all([inputfiles], **kwargs)
    list_of_xyz, labels = [], []
    for inputfile in inputfiles:
        base = _basename(inputfile)
        xyz_data = load_xyz(inputfile, sanity_check=True, center_COM=True)
        xyz_bulk, xyz_ligs = get_core_atoms_and_outer_layer(xyz_data, surf_type='cap')
        _, xyz_surf = get_core_atoms_and_outer_layer(xyz_bulk, surf_type='nc')
        list_of_xyz.extend([xyz_bulk, xyz_surf, xyz_ligs])
        labels.extend([f"{base} | core", f"{base} | surf", f"{base} | ligands"])
    return _plot_py3Dmol(list_of_xyz, labels=labels, **kwargs)


def visualize_local_frame(cluster, r0, ex, ey, ez, scale=1.5):
    """
    cluster : Nx4 array, columns = [label, x, y, z]
    r0      : origin of the frame (xyz, float)
    ex, ey, ez : unit vectors of the frame
    scale   : length of the displayed arrows
    """
    view = py3Dmol.view(width=600, height=500)

    # Draw cluster atoms as spheres
    xyz_block = f"{len(cluster)}\ncluster\n"
    for atom in cluster:
        xyz_block += f"{atom[0]}  {atom[1]}  {atom[2]}  {atom[3]}\n"
    view.addModel(xyz_block, "xyz")
    view.setStyle({"sphere": {"scale": 0.3}})
    view.addStyle({"stick": {"radius": 0.1}})

    # Draw frame axes as cylinders + cones (arrows)
    colors = {"ex": "red", "ey": "green", "ez": "blue"}
    axes  = {"ex": np.array(ex, dtype=float),
             "ey": np.array(ey, dtype=float),   
             "ez": np.array(ez, dtype=float)
            }

    for name, vec in axes.items():
        tip = np.array(r0, dtype=float) + scale * vec
        # Cylinder (shaft)
        view.addCylinder({
            "start": {"x": float(r0[0]),  "y": float(r0[1]),  "z": float(r0[2])},
            "end":   {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
            "radius": 0.08,
            "color": colors[name],
            "fromCap": True, "toCap": False
        })
        # Cone (arrowhead)
        cone_base = r0 + 0.85 * scale * vec
        view.addCylinder({
            "start": {"x": float(cone_base[0]), "y": float(cone_base[1]), "z": float(cone_base[2])},
            "end":   {"x": float(tip[0]),       "y": float(tip[1]),       "z": float(tip[2])},
            "radius": 0.18,
            "color": colors[name],
            "fromCap": False, "toCap": True
        })
        # Label
        view.addLabel(name, {
            "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
            "fontColor": colors[name],
            "backgroundColor": "white",
            "fontSize": 14
        })

    # Mark the origin r0
    view.addSphere({
        "center": {"x": float(r0[0]), "y": float(r0[1]), "z": float(r0[2])},
        "radius": 0.25,
        "color": "yellow"
    })

    view.zoomTo()
    view.show()
    return view
