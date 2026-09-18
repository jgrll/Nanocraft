# Facets module
from .viz import _basename
from ..nanocraft.ioxyz import load_xyz
from ..nanocraft.srflyrs import surface_tessellation
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _merge_simplices(simplices_pos, center, threshold=20.0, plane_tol=1e-3, vertex_tol=1e-4):
    simplices_pos = np.asarray(simplices_pos, dtype=float)
    center = np.asarray(center, dtype=float)

    nsimp = len(simplices_pos)
    if nsimp == 0:
        return []

    def _vertex_key(v):
        return tuple(np.round(v / vertex_tol).astype(int))

    # 1. Compute outward-oriented normals and plane offsets
    normals = np.zeros((nsimp, 3), dtype=float)
    offsets = np.zeros(nsimp, dtype=float)

    for i, tri in enumerate(simplices_pos):
        u = tri[1] - tri[0]
        v = tri[2] - tri[0]
        n = np.cross(u, v)
        norm = np.linalg.norm(n)

        if norm < 1e-12:
            raise ValueError(f"Degenerate simplex at index {i}")

        n /= norm

        if np.dot(n, tri[0] - center) < 0:
            n = -n

        normals[i] = n
        offsets[i] = np.dot(n, tri[0])

    # 2. Build adjacency graph between neighboring coplanar triangles
    adjacency = [[] for _ in range(nsimp)]
    tri_vertex_sets = [{_vertex_key(v) for v in tri} for tri in simplices_pos]

    for i in range(nsimp):
        for j in range(i + 1, nsimp):
            common = tri_vertex_sets[i].intersection(tri_vertex_sets[j])
            if len(common) < 2:
                continue

            dot = np.dot(normals[i], normals[j])
            angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
            same_plane = abs(offsets[i] - offsets[j]) <= plane_tol

            if angle <= threshold and same_plane:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # 3. Connected components = triangle patches
    visited = np.zeros(nsimp, dtype=bool)
    patches = []

    for start in range(nsimp):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        patch = []

        while stack:
            current = stack.pop()
            patch.append(current)

            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        patches.append(patch)

    # 4. Convert each patch of triangles into one boundary polygon
    polygons = []

    for patch in patches:
        edge_count = {}
        vertex_map = {}

        for tri_idx in patch:
            tri = simplices_pos[tri_idx]
            keys = [_vertex_key(v) for v in tri]

            for k, v in zip(keys, tri):
                vertex_map[k] = v

            tri_edges = [
                (keys[0], keys[1]),
                (keys[1], keys[2]),
                (keys[2], keys[0]),
            ]

            for a, b in tri_edges:
                edge = tuple(sorted((a, b)))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        if len(boundary_edges) < 3:
            continue

        # Build boundary adjacency
        boundary_adj = {}
        for a, b in boundary_edges:
            boundary_adj.setdefault(a, []).append(b)
            boundary_adj.setdefault(b, []).append(a)

        # Walk boundary loop
        start = boundary_edges[0][0]
        polygon_keys = [start]
        prev = None
        current = start

        while True:
            neighbors = boundary_adj[current]

            if prev is None:
                nxt = neighbors[0]
            else:
                if len(neighbors) == 1:
                    nxt = neighbors[0]
                else:
                    nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]

            if nxt == start:
                break

            polygon_keys.append(nxt)
            prev, current = current, nxt

            if len(polygon_keys) > len(boundary_adj) + 5:
                raise RuntimeError("Boundary walk failed; patch may be non-manifold.")

        polygon = np.array([vertex_map[k] for k in polygon_keys], dtype=float)

        # Optional: sort vertices consistently in polygon plane
        centroid = polygon.mean(axis=0)
        normal = normals[patch[0]]

        ref = polygon[0] - centroid
        ref /= np.linalg.norm(ref)

        ref2 = np.cross(normal, ref)
        ref2 /= np.linalg.norm(ref2)

        rel = polygon - centroid
        angles = np.arctan2(rel @ ref2, rel @ ref)
        order = np.argsort(angles)

        polygon = polygon[order]
        polygons.append(polygon)

    return polygons


def visualize_simplices(inputfiles, method='hull', alpha=1.2, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_simplices([inputfiles], method, alpha, **kwargs)
    list_of_polygons, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=False, center_COM=True)
        _, simplices, _, _  = surface_tessellation(xyz_data, method=method, alpha=alpha)
        list_of_polygons.append(simplices)
        labels.append(_basename(inputfile))
    return _plot_plotly_simplices(list_of_polygons, labels=labels, **kwargs)


def visualize_patches(inputfiles, method='hull', alpha=1.2, threshold=10.0, **kwargs):
    if not isinstance(inputfiles, list):
        return visualize_patches([inputfiles], method, alpha, threshold, **kwargs)
    list_of_polygons, labels = [], []
    for inputfile in inputfiles:
        xyz_data = load_xyz(inputfile, sanity_check=False, center_COM=True)
        _, simplices, _, _  = surface_tessellation(xyz_data, method=method, alpha=alpha)
        patches = _merge_simplices(simplices, center=[0.0, 0.0, 0.0], threshold=threshold)
        list_of_polygons.append(patches)
        labels.append(_basename(inputfile))
    return _plot_plotly_patches(list_of_polygons, labels=labels, **kwargs)


def _plot_plotly_simplices(list_of_polygons, labels=None, ncols=5, style="default", 
                 color_scheme="grey", width=900, height=250, background="white", spin=False):
    
    n_struct = len(list_of_polygons)
    if n_struct == 0:
        raise ValueError("list_of_polygons is empty")

    nrows = (n_struct + ncols - 1) // ncols

    if labels is None:
        labels = [f"structure {i+1}" for i in range(n_struct)]

    if len(labels) != n_struct:
        raise ValueError("labels must have the same length as list_of_polygons")

    specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]

    subplot_titles = labels + [""] * (nrows * ncols - n_struct)

    fig = make_subplots(rows=nrows, cols=ncols, specs=specs,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.03, vertical_spacing=0.08,)

    def triangulate_fan(m):
        i_idx, j_idx, k_idx = [], [], []
        for t in range(1, m - 1):
            i_idx.append(0)
            j_idx.append(t)
            k_idx.append(t + 1)
        return i_idx, j_idx, k_idx

    for idx, polyhedron in enumerate(list_of_polygons):
        row = idx // ncols + 1
        col = idx % ncols + 1

        polyhedron = np.asarray(polyhedron)

        if polyhedron.ndim != 3 or polyhedron.shape[2] != 3:
            raise ValueError(
                f"Each entry must have shape (N, M, 3). Got {polyhedron.shape} at index {idx}."
            )

        for polygon in polyhedron:
            polygon = np.asarray(polygon)
            m = polygon.shape[0]
            if m < 3:
                continue
            x, y, z = polygon[:, 0], polygon[:, 1], polygon[:, 2]

            if style in ("default", "solid"):
                ii, jj, kk = triangulate_fan(m)
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z, i=ii, j=jj, k=kk,
                    color=color_scheme, opacity=0.55 if style == "default" else 0.85,
                    flatshading=True, hoverinfo="skip", showscale=False), row=row, col=col,)
            if style in ("default", "wireframe", "solid"):
                xe = np.append(x, x[0])
                ye = np.append(y, y[0])
                ze = np.append(z, z[0])
                fig.add_trace(go.Scatter3d(
                    x=xe, y=ye, z=ze,
                    mode="lines", line=dict(color="black", width=4), 
                    hoverinfo="skip", showlegend=False,),
                    row=row,col=col,)
            if style == "points":
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="markers", marker=dict(size=4, color=color_scheme),
                    hoverinfo="skip", showlegend=False), row=row, col=col)
                
        scene_name = "scene" if idx == 0 else f"scene{idx+1}"

        fig.layout[scene_name].update(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=background,
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),)
        
    fig.update_layout(
        width=width,
        height=height * nrows,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    # fig.show()
    return fig



def _plot_plotly_patches(list_of_polygons, labels=None, ncols=5, style="default",
    color_scheme="grey", width=900, height=250, background="white", spin=False):
    
    n_struct = len(list_of_polygons)
    if n_struct == 0:
        raise ValueError("list_of_polygons is empty")

    nrows = (n_struct + ncols - 1) // ncols

    if labels is None:
        labels = [f"structure {i + 1}" for i in range(n_struct)]

    if len(labels) != n_struct:
        raise ValueError("labels must have the same length as list_of_polygons")

    specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]
    subplot_titles = labels + [""] * (nrows * ncols - n_struct)

    fig = make_subplots(rows=nrows, cols=ncols,
                        specs=specs, subplot_titles=subplot_titles,
                        horizontal_spacing=0.03, vertical_spacing=0.08)

    def triangulate_fan(m):
        i_idx, j_idx, k_idx = [], [], []
        for t in range(1, m - 1):
            i_idx.append(0)
            j_idx.append(t)
            k_idx.append(t + 1)
        return i_idx, j_idx, k_idx

    for idx, polyhedron in enumerate(list_of_polygons):
        row = idx // ncols + 1
        col = idx % ncols + 1

        if not isinstance(polyhedron, (list, tuple)):
            raise ValueError(
                f"Each structure must be a list of polygons. Got {type(polyhedron)} at index {idx}."
            )

        for polygon in polyhedron:
            polygon = np.asarray(polygon, dtype=float)

            if polygon.ndim != 2 or polygon.shape[1] != 3:
                raise ValueError(
                    f"Each polygon must have shape (M, 3). Got shape {polygon.shape}."
                )

            m = polygon.shape[0]
            if m < 3:
                continue

            x, y, z = polygon[:, 0], polygon[:, 1], polygon[:, 2]

            if style in ("default", "solid"):
                ii, jj, kk = triangulate_fan(m)
                fig.add_trace(
                    go.Mesh3d(x=x, y=y, z=z,
                              i=ii, j=jj, k=kk, color=color_scheme,
                              opacity=0.55 if style == "default" else 0.85,
                              flatshading=True, hoverinfo="skip", showscale=False),
                              row=row,col=col,
                              )

            if style in ("default", "wireframe", "solid"):
                xe, ye, ze = np.r_[x, x[0]], np.r_[y, y[0]], np.r_[z, z[0]]

                fig.add_trace(
                    go.Scatter3d(
                        x=xe, y=ye, z=ze,
                        mode="lines", line=dict(color="black", width=4),
                        hoverinfo="skip", showlegend=False), row=row, col=col,
                        )

            if style == "points":
                fig.add_trace(
                    go.Scatter3d(x=x, y=y, z=z,
                                 mode="markers", marker=dict(size=4, color=color_scheme),
                                 hoverinfo="skip", showlegend=False), row=row, col=col
                                 )

        scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
        fig.layout[scene_name].update(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=background,
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        )

    fig.update_layout(
        width=width,
        height=height * nrows,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )

    return fig