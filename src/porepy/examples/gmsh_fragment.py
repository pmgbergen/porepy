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
    frac_4 = add_surface(
        [
            np.array([-0.2, -0, -0]),
            np.array([1, 0.0, -0]),
            np.array([1, 1, 1]),
            np.array([-0.2, 1, 1]),
        ]
    )

    fac.synchronize()

    fractures = [(2, frac_1), (2, frac_2), (2, frac_3), (2, frac_4)]

    _, isect_mapping = fac.fragment(
        fractures, [(3, domain)], removeObject=True, removeTool=True
    )

    fac.synchronize()

    # Partial implementation. Intersection lines are either on the boundary or embedded
    # in fractures. Make a list of both.
    bnd_lines = []
    embedded_lines = []
    # Challenge: Since the mdg graph only accepts single edges between node pairs
    # (subdomains), if two intersection lines cross (think a Rubik's cube geometry), the
    # split part of the intersection line must be assigned the same physical tag, thus
    # generate a single subdomain grid. Achieving this will take some work; for
    # starters, keep track of the fracture indices that gave rise to each line.
    fi_bnd = []
    fi_embedded = []

    # Loop over all identified fragments of the fractures, find their boundary and
    # embedded lines.
    #
    # TODO: What if two fractures intersect in a point? This is likely not covered here,
    # and not considered in the current implementation of md dynamics in general.
    for fi, new_frac in enumerate(isect_mapping):
        # A fracture can be split into multiple sub-fractures if they are fully cut by
        # other fractures.
        for subfrac in new_frac:
            if subfrac[0] == 3:  # This is the domain.
                continue
            # Get the boundary of the sub-fracture. It can contain both lines on the
            # boundary of the original fracture and lines on the boundary of
            # subfractures that were introduced because a fracture was cut in two.
            bnd = gmsh.model.get_boundary([subfrac])
            # Loop over the boundary.
            for parent_map in bnd:
                if (
                    parent_map[0] == 1
                ):  # This is a line, not a point (would be b[0] == 0).
                    bnd_lines.append(parent_map[1])
                    # Keep track of the fracture index for each boundary line. Using fi
                    # (the enumeration counter of the outer for loop) ensures that even
                    # if a fracture was split into two sub-fractures during
                    # fragmentation, they will still be associated with the original
                    # fracture index.
                    fi_bnd.append(fi)

            # Also find lines that are embedded in this subfracture (this will be an
            # intersection line that does not cut subfrac in two).
            embedded = gmsh.model.mesh.get_embedded(*subfrac)
            for line in embedded:
                if line[0] == 1:
                    embedded_lines.append(line[1])
                    # Also keep track of the fracture index for each embedded line.
                    fi_embedded.append(fi)

    # For a boundary line to be an intersection, it must be shared by at least two
    # fractures. TODO: What if it is on the boundary of one, but not the other, in a
    # T-style intersection? Should be embedded in the other then? EK thinks this should
    # be the case. Similarly, L-style intersection (boundary on both) should be picked
    # up here.
    num_lines_occ = np.bincount(np.abs(bnd_lines))
    # Find the 'interesting' boundary lines, i.e. those occuring more than once.
    boundary_lines = np.where(num_lines_occ > 1)[0]
    all_lines = np.hstack((embedded_lines, boundary_lines))
    # Fracture intersection lines, to be added as physical lines.
    intersection_lines = np.unique(all_lines)

    # Now, we need to find which intersection lines stem from the same set of
    # intersecting fractures (this can be two or more fractures). This requires some
    # juggling of indices.

    # Turn the fracture indices into numpy arrays.
    fi_bnd = np.array(fi_bnd)
    fi_embedded = np.array(fi_embedded)
    # For each intersection line, this will be a list of its parent fractures.
    line_parents = []
    for line in intersection_lines:
        # Find the set of parents, looking at both boundary and embedded lines (an
        # intersection can be on the boundary of fracture, but not the other).
        parent = np.hstack(
            (
                fi_bnd[np.where(np.abs(bnd_lines) == line)[0]],
                fi_embedded[np.where(np.abs(embedded_lines) == line)[0]],
            )
        )
        # Uniquify (thereby also sort) and turn to list.
        line_parents.append(np.unique(parent).tolist())

    # Now we need to find the unique parent sets. Since line_parents can have a varying
    # number of elements, we cannot just do a numpy unique, but instead need to process
    # each number of parents separately (if the number of parents differs, clearly, the
    # sets of parents must also differ).

    # Do a count.
    num_parents = np.array([len(lp) for lp in line_parents])
    # This will be the mapping from line indices to their parent set indices.
    parent_of_intersection_lines = np.full(num_parents.size, -1)

    # Counter over intersection lines. Linked to the physical tags that will be
    # associated with the intersection lines. Note that there is no requirement that
    # this is related to the physical tag of the parent fracture (to the degree we care
    # about such intersections, we go through the grid information generated by gmsh).
    num_line_parent_counter = 0

    # Loop over all unique parent counts.
    for n in np.unique(num_parents):
        # Find all intersection lines that has this number of parents.
        inds = np.where(num_parents == n)[0]
        # Find the unique number of parents and the map from all intersection lines with
        # 'n' parents to the unique set.
        unique_parent, parent_map = np.unique(
            [line_parents[i] for i in inds], axis=0, return_inverse=True
        )
        # Store the parent identification for this set of intersection lines.
        parent_of_intersection_lines[inds] = parent_map + num_line_parent_counter
        # Increase the counter.
        num_line_parent_counter += unique_parent.shape[0]
    # Done with the intersection line processing.

    # Find intersection points: These by definition lie on the boundary of intersection
    # lines, so we loop over the latter, store their boundary points and identify which
    # points occur more than once.u
    points_of_intersection_lines = []
    for line in intersection_lines:
        for bp in gmsh.model.get_boundary([(1, line)]):
            points_of_intersection_lines.append(bp[1])

    num_point_occ = np.bincount(points_of_intersection_lines)
    intersection_points = np.where(num_point_occ > 1)[0]

    ## Export physical entities to gmsh.

    # Intersection points.
    for i in intersection_points:
        gmsh.model.addPhysicalGroup(0, [i], -1, f"FRACTURE_INTERSECTION_POINT_{i}")
    # Intersection lines.
    for li in range(num_line_parent_counter):
        this_parent = np.where(parent_of_intersection_lines == li)[0]

        gmsh.model.addPhysicalGroup(
            1,
            intersection_lines[this_parent].tolist(),
            -1,
            f"FRACTURE_INTERSECTION_LINE_{li}",
        )
    # Fractures.
    for i, frac in enumerate(isect_mapping):
        subfracs = []
        for subfrac in frac:
            if subfrac[0] == 2:
                subfracs.append(subfrac[1])
        if subfracs:
            gmsh.model.addPhysicalGroup(2, subfracs, -1, f"FRACTURE_{i}")
    # The domain.
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
        truncated_line, parent_map = fac.intersect(
            [line], [(2, domain)], removeTool=False, removeObject=False
        )
        if len(truncated_line) > 0:
            # The line was partly outside the domain. It must be replaced.
            new_lines[ind] = truncated_line[0]
        elif len(parent_map[0]) == 0:
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
