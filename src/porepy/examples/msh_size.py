import numpy as np
import gmsh
import porepy as pp

# Mesh and domain definition
gmsh.initialize()
factory = gmsh.model.occ
dx = 1
factory.synchronize()


def fracture_generator(ind: int) -> list[pp.LineFracture]:
    match ind:
        case 0:
            # Isolated fracture.
            return [pp.LineFracture(np.array([[0.1, 0.3], [0.1, 0.3]]))]
        case 1:
            # Fracture that crosses the domain.
            return [pp.LineFracture(np.array([[-0.2, 0.2], [0.5, 0.5]]))]
        case 2:
            # Fracture that is close to the boundary.
            return [pp.LineFracture(np.array([[0.5, 0.5], [0.8, 0.97]]))]
        case 3:
            # Short fracture.
            return [pp.LineFracture(np.array([[0.3, 0.3], [0.75, 0.77]]))]
        case 4:
            # Fractures crossing in a nice angle.
            return [
                pp.LineFracture(np.array([[0.7, 0.9], [0.2, 0.2]])),
                pp.LineFracture(np.array([[0.8, 0.8], [0.1, 0.3]])),
            ]
        case 5:
            # Fractures crossing in a small angle.
            return [
                pp.LineFracture(np.array([[0.7, 0.9], [0.4, 0.4]])),
                pp.LineFracture(np.array([[0.7, 0.9], [0.38, 0.42]])),
            ]
        case 6:
            # Parallell fractures
            return [
                pp.LineFracture(np.array([[0.7, 0.9], [0.59, 0.59]])),
                pp.LineFracture(np.array([[0.7, 0.9], [0.61, 0.61]])),
            ]
        case 7:
            # Almost parallel fractures.
            return [
                pp.LineFracture(np.array([[0.6, 0.95], [0.79, 0.79]])),
                pp.LineFracture(np.array([[0.75, 0.85], [0.81, 0.815]])),
            ]
        case _:
            raise ValueError("Invalid fracture index")


fractures = [f for i in range(8) for f in fracture_generator(i)]
network = pp.create_fracture_network(fractures, domain=pp.domains.nd_cube_domain(2, dx))
mesh_size = {"mesh_size_frac": 0.01, "mesh_size_bound": 0.1}
mdg = network.mesh(mesh_args=mesh_size)
