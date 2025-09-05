import gmsh

import meshio

import porepy as pp
import numpy as np

run_3d = True
gmsh.initialize()

fac = gmsh.model.occ

if run_3d:
    domain = fac.add_box(-1, -1, -1, 4, 4, 4)

    def add_surface(pts):
        num_pts = len(pts)
        gmsh_pts = [fac.addPoint(*pt) for pt in pts]
        pt_indices = np.concatenate([np.arange(num_pts), [0]])
        gmsh_lines = [
            fac.add_line(gmsh_pts[pt_indices[i]], gmsh_pts[pt_indices[i + 1]])
            for i in range(num_pts)
        ]
        loop = fac.add_curve_loop(gmsh_lines)
        return fac.add_plane_surface([loop])

    frac_1 = add_surface(
        [
            np.array([0, 0, 0.5]),
            np.array([2, 0, 0.5]),
            np.array([2, 1, 0.5]),
            np.array([0, 1, 0.5]),
        ]
    )
    frac_2 = add_surface(
        [
            np.array([-0.5, 0.5, 0]),
            np.array([1, 0.5, 0]),
            np.array([1, 0.5, 1]),
            np.array([-0.5, 0.5, 1]),
        ]
    )
    frac_3 = add_surface(
        [
            np.array([0.5, 0, 0]),
            np.array([0.5, 1, 0]),
            np.array([0.5, 1, 1]),
            np.array([0.5, 0, 1]),
        ]
    )

    fac.synchronize()

    fractures = [(2, frac_1), (2, frac_2), (2, frac_3)]

    _, isect_mapping = fac.fragment(
        fractures, [(3, domain)], removeObject=True, removeTool=True
    )

    fac.synchronize()

    # Partial implementation.
    # Intersection lines are either on the boundary or embedded in fractures. Make a list of both.
    bnd_lines = []
    embedded_lines = []
    # Problem: Since the mdg graph only accepts single edges between node pairs
    # (subdomains), if two intersection lines cross (think a Rubik's cube geometry), the
    # split part of the intersection line must be assigned the same physical tag, thus
    # generate a single subdomain grid. For starters, keep track of the fracture indices
    # that gave rise to each line.
    fi_bnd = []
    fi_embedded = []

    for fi, new_frac in enumerate(isect_mapping):
        for subfrac in new_frac:
            if subfrac[0] == 3:  # This is the domain.
                continue
            bnd = gmsh.model.get_boundary([subfrac])
            for b in bnd:
                if b[0] == 1:
                    bnd_lines.append(b[1])
                    # Keep track of the fracture index for each boundary line. Using fi
                    # (the enumeration counter) here means that even if a fracture was
                    # split into two sub-fractures during fragmentation, they will still
                    # be associated with the original fracture index.
                    fi_bnd.append(fi)

            embedded = gmsh.model.mesh.get_embedded(*subfrac)
            for line in embedded:
                if line[0] == 1:
                    embedded_lines.append(line[1])
                    fi_embedded.append(fi)

    # For a boundary line to be an intersection, it must be shared by at least two
    # fractures. TODO: What if it is on the boundary of one, but not the other? Should
    # be embedded in the other then?
    num_lines_occ = np.bincount(np.abs(bnd_lines))
    boundary_lines = np.where(num_lines_occ > 1)[0]
    all_lines = np.hstack((embedded_lines, boundary_lines))
    # Fracture intersection lines, to be added as physical lines.
    intersection_lines = np.unique(all_lines)

    fi_bnd = np.array(fi_bnd)
    fi_embedded = np.array(fi_embedded)

    line_parents = []

    # This is an attempt to identify pairs of parents that give rise to a line. It will
    # likely break if three fractures meet along a single line (then the elements
    # line_parents will not have the same size, hence the unique on line_parents will
    # break. Perhaps we can do a unique for each size of the parent set?
    for line in intersection_lines:
        parent = np.hstack(
            (
                fi_bnd[np.where(np.abs(bnd_lines) == line)[0]],
                fi_embedded[np.where(np.abs(embedded_lines) == line)[0]],
            )
        )
        line_parents.append(np.unique(parent).tolist())
    _, intersection_line_to_fracture_pair = np.unique(
        line_parents, axis=0, return_inverse=True
    )

    points_of_intersection_lines = []

    for line in intersection_lines:
        for bp in gmsh.model.get_boundary([(1, line)]):
            points_of_intersection_lines.append(bp[1])

    num_point_occ = np.bincount(points_of_intersection_lines)
    intersection_points = np.where(num_point_occ > 1)[0]

    debug = []

    for i in intersection_points:
        gmsh.model.addPhysicalGroup(0, [i], -1, f"FRACTURE_INTERSECTION_POINT_{i}")

    for li, line in enumerate(line_parents):
        gmsh.model.addPhysicalGroup(1, line, -1, f"FRACTURE_INTERSECTION_LINE_{li}")

    for i, frac in enumerate(isect_mapping):
        subfracs = []
        for subfrac in frac:
            if subfrac[0] == 2:
                subfracs.append(subfrac[1])
        if subfracs:
            gmsh.model.addPhysicalGroup(2, subfracs, -1, f"FRACTURE_{i}")

    gmsh.model.addPhysicalGroup(3, [domain], -1, "DOMAIN")

    fac.synchronize()

    gmsh.write("tmp.geo_unrolled")

    fn = "tmp.msh"

    # Create a gmsh mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(fn)

    msh = meshio.read(fn)

    mdg = pp.fracture_importer.dfm_from_gmsh(fn, dim=3)
    print(mdg)

    debug = []


else:  # 2D
    # Define the domain. This will normally be the task of the FractureNetwork class.
    domain = fac.add_rectangle(-1, -1, 0, 4, 4)

    # Add a few fractures. This will normally be the task of the individual Fracture objects.

    # Create a line [x1, y1, z1], [x2, y2, z2]
    p_0 = fac.addPoint(0, 0, 0)
    p_1 = fac.addPoint(1, 1, 0)
    line1 = fac.add_line(p_0, p_1)

    # Create a new line crossing the first one
    p_2 = fac.addPoint(0, 1, 0)
    p_3 = fac.addPoint(1, 0.5, 0)
    line2 = fac.add_line(p_2, p_3)

    # Add a third line which crosses the first line, but not the second.
    p_4 = fac.addPoint(0.5, -0.8, 0)
    p_5 = fac.addPoint(0.5, -0, 0)
    line3 = fac.add_line(p_4, p_5)

    # List of fractures. This is roughly the expected output after a FractureNetwork object
    # has looped over its fractures and had them represented in the gmsh.occ kernel.
    lines = [(1, line1), (1, line2), (1, line3)]

    # Synchronize the GMSH model before fragmenting.
    fac.synchronize()

    ## This part identifies fractures that are fully outside the domain and kicks them out.

    new_lines = {}
    lines_removed = []

    for ind, line in enumerate(lines):
        truncated_line, b = fac.intersect(
            [line], [(2, domain)], removeTool=False, removeObject=False
        )
        if len(truncated_line) > 0:
            # The line was partly outside the domain. It must be replaced.
            new_lines[ind] = truncated_line[0]
        elif len(b[0]) == 0:
            # The line was fully outside the domain. It will be deleted.
            lines_removed.append(ind)

    # Remove the lines from the gmsh representation. Recursive is critical here, or else the
    # boundary of 'line' will continue to be present.
    for line in lines_removed:
        fac.remove([lines[line]], recursive=True)

    for old_line, new_line in new_lines.items():
        fac.remove([lines[old_line]], recursive=True)
        lines[old_line] = new_line

    lines = [lines[i] for i in range(len(lines)) if i not in lines_removed]

    fac.synchronize()

    # Make gmsh calculate the intersections between the lines, using the domain as a
    # secondary object (the latter will by magic ensure that the lines are embedded in the
    # domain, hence the mesh will conform to the fractures). The removal statements here
    # will replace the old (possibly intersecting) lines with new, split lines. Similarly,
    # the removal of the domain (removeTool, no idea why) avoids the domain being present
    # twice.
    _, isect_mapping = fac.fragment(
        lines, [(2, domain)], removeObject=True, removeTool=True
    )

    fac.synchronize()

    # During intersection removal, gmsh will add intersection points and replace the lines
    # with non-intersecting polylines (example: Two lines intersecting as a cross become
    # four lines with a common point). We need to identify these intersecting points.
    # Annoyingly, gmsh also reassigns tags during fragmentation. Hence we need to find
    # intersection points on our own, as is done below.
    intersection_points = []
    # Loop over the mappings from old lines to new segments. The idea is to find the
    # boundary points of all segments, identify those that occur more than once - these will
    # be intersections - and store the tag that gmsh has assigned them.
    for old_line in isect_mapping:
        all_boundary_points_of_segments = []
        for segment in old_line:
            all_boundary_points_of_segments += gmsh.model.get_boundary([segment])

        # Find the unique boundary points and obtain a mapping from the full set of boundary
        # points to the unique ones.
        unique_boundary_points, u2a_ind = np.unique(
            all_boundary_points_of_segments, axis=0, return_index=True
        )
        # Count the number of occurrences of each unique boundary point. Those that occur
        # more than once will be intersections.
        multiple_occs = np.where(np.bincount(u2a_ind) > 1)[0]
        # Store the gmsh representation of the intersection points.
        intersection_points += [unique_boundary_points[i] for i in multiple_occs]
    # Finally, we need to uniquify the intersection points, since the same point will have
    # been identified in at least two old lines.
    unique_intersection_points = np.unique(np.vstack(intersection_points), axis=0)

    # Represent the intersection points, fractures and domain as physical points.
    fac.synchronize()

    # Intersection points can be dealt with right away.
    for i, pt in enumerate(unique_intersection_points):
        gmsh.model.addPhysicalGroup(0, [pt[1]], -1, f"FRACTURE_INTERSECTION_POINT_{i}")

    fac.synchronize()

    # Since fractures may have been split at intersection points, we need to collect all the
    # segments (found in isect_mapping) into a single physical group.
    for i, line_group in enumerate(isect_mapping):
        all_lines = []
        for line in line_group:
            if line[0] == 1:
                all_lines.append(line[1])
        if all_lines:
            gmsh.model.addPhysicalGroup(1, all_lines, -1, f"FRACTURE_{i}")

    gmsh.model.addPhysicalGroup(2, [domain], -1, "DOMAIN")

    fac.synchronize()

    gmsh.write("tmp.geo_unrolled")

    fn = "tmp.msh"

    # Create a gmsh mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(fn)

    msh = meshio.read(fn)

    mdg = pp.fracture_importer.dfm_from_gmsh(fn, dim=2)
    print(mdg)

    debug = []
