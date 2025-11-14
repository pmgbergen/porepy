import porepy as pp
import numpy as np


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
            return [pp.LineFracture(np.array([[0.2, 0.2], [0.35, 0.40]]))]
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
        case 8:
            # Almost a T-intersection.
            return [
                pp.LineFracture(np.array([[0.5, 0.5], [0.61, 0.7]])),
                pp.LineFracture(np.array([[0.4, 0.6], [0.6, 0.6]])),
            ]
        case 9:
            # A T-intersection.
            return [
                pp.LineFracture(np.array([[0.5, 0.5], [0.45, 0.55]])),
                pp.LineFracture(np.array([[0.4, 0.6], [0.45, 0.45]])),
            ]
        case 10:
            # X-intersection which is almost a T-intersection.
            return [
                pp.LineFracture(np.array([[0.5, 0.5], [0.34, 0.40]])),
                pp.LineFracture(np.array([[0.4, 0.6], [0.35, 0.35]])),
            ]
        case 11:
            # Three almost parallel fractures, but with little overlap.
            return [
                pp.LineFracture(np.array([[0.2, 0.4], [0.1, 0.1]])),
                pp.LineFracture(np.array([[0.38, 0.50], [0.12, 0.12]])),
                pp.LineFracture(np.array([[0.47, 0.70], [0.1, 0.1]])),
            ]
        case 12:
            # Fracture almost hitting the boundary, orthorgonal to the boundary segment.
            return [pp.LineFracture(np.array([[0.01, 0.10], [0.2, 0.2]]))]
        case 13:
            # Fracture parallel to boundary, very close.
            return [pp.LineFracture(np.array([[0.7, 0.9], [0.03, 0.03]]))]
        case 14:
            # Almost L-type intersection.
            return [
                pp.LineFracture(np.array([[0.1, 0.25], [0.62, 0.62]])),
                pp.LineFracture(np.array([[0.24, 0.24], [0.64, 0.75]])),
            ]
        case _:
            raise ValueError("Invalid fracture index")


benchmark = False

if benchmark:
    fractures = pp.fracture_sets.benchmark_2d_case_4()  # [:50]
    domain = pp.Domain({"xmin": 0, "xmax": 700, "ymin": 0, "ymax": 600})
    network = pp.create_fracture_network(fractures, domain=domain)
    mesh_size = {"mesh_size_frac": 10.0, "mesh_size_bound": 100.0}
    mdg = network.mesh(mesh_args=mesh_size)
else:
    fractures = [f for i in range(15) for f in fracture_generator(i)]
    # fractures = [f for i in [8, 12] for f in fracture_generator(i)]
    network = pp.create_fracture_network(
        fractures, domain=pp.domains.nd_cube_domain(2, 1)
    )
    mesh_size = {"mesh_size_frac": 0.05, "mesh_size_bound": 0.1}
    mdg = network.mesh(mesh_args=mesh_size)

pp.plot_grid(mdg, figsize=(10, 8), linewidth=0.2, plot_2d=True)
print(mdg)
