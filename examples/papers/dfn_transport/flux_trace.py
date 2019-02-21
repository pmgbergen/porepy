def jump_flux(gb, flux_mortar):

    # loop on the lagrange multiplier nodes
    for g in gb.grids_of_dimension(1):

        # loop on the associated edges
        for _, d_e in gb.edges_of_node(g):

            # get the projector from the mortar grid to the slave
            proj = d_e["mortar_grid"].mortar_to_slave_int()

            # project the mortar variable from the mortar grid to the 1d
            # grid by thus doing the sum on the same 1d-cell
            # the variable now contains the jump of the flux from the current 2d
            # fracture to thourgh the 1d object.
            jump_flux = proj.dot(d_e[flux_mortar])

            print(jump_flux)

        print("----")
