"""
Testing the functionality related to model geometries. There functions are covered:
- SubsurfaceCuboidDomain
- TwoWells3d
- TwoEllipticFractures3d

"""

import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SubsurfaceCuboidDomain,
    TwoEllipticFractures3d,
    TwoWells3d,
)
from porepy.applications.test_utils.models import NoPhysics


class SubsurfaceDomainModel(SubsurfaceCuboidDomain):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_subsurface_set_domain():
    """
    Check whether the domain in x, y, and z directions are correctly created from
    the given domain size.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = SubsurfaceDomainModel(params)
    model.set_domain()
    box = model._domain.bounding_box

    assert box["xmin"] == 0.0
    assert box["xmax"] == 10.0
    assert box["ymin"] == 0.0
    assert box["ymax"] == 20.0
    assert box["zmin"] == -30.0
    assert box["zmax"] == 0.0


class TwoWells3dModel(TwoWells3d):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_created_well_network():
    """
    Check whether the expected well network for the model are correctly created. This
    test specifically checks that two wells are exactly created and assigned with the
    corresponding well names.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoWells3dModel(params)
    model.set_domain()
    model.set_wells()

    wells = model._wells
    names = [well.tags["well_name"] for well in wells]

    assert len(wells) == 2
    assert names == ["injection_well", "production_well"]


class TwoEllipticFractures3dModel(TwoEllipticFractures3d):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_created_elliptic_fracs():
    """
    Check whether elliptic fractures are correctly created and parameterized.
    This test specifically check that two elliptic fractures are exactly created
    with expected major axis length.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoEllipticFractures3dModel(params)
    model.set_domain()
    model.set_fractures()
    fractures = model._fractures
    expected_major_axis = 0.2 * 10.0

    assert len(fractures) == 2
    assert np.allclose(model.fracture_major_axes, expected_major_axis)


class CartesianSubsurfaceModel(SubsurfaceCuboidDomain, NoPhysics):  # type: ignore[misc]
    """A subsurface model with two crossing, axis-aligned fractures.

    The fractures are compatible with structured meshing, in contrast to the elliptic
    fractures of :class:`TwoEllipticFractures3d`.

    """

    def set_fractures(self) -> None:
        self._fractures = [
            pp.PlaneFracture(
                np.array(
                    [
                        [5.0, 5.0, 5.0, 5.0],
                        [5.0, 15.0, 15.0, 5.0],
                        [-20.0, -20.0, -10.0, -10.0],
                    ]
                )
            ),
            pp.PlaneFracture(
                np.array(
                    [
                        [0.0, 10.0, 10.0, 0.0],
                        [5.0, 5.0, 15.0, 15.0],
                        [-15.0, -15.0, -15.0, -15.0],
                    ]
                )
            ),
        ]


def test_cartesian_meshing_of_subsurface_domain():
    """Check that a cuboid subsurface domain can be meshed with Cartesian grids.

    Such a domain extends downwards from the surface, hence its lower corner is not in
    the origin. Failure indicates that the structured meshing does not account for the
    position of the domain, see pp.create_mdg.

    """
    domain_sizes = np.array([10.0, 20.0, 30.0])
    model = CartesianSubsurfaceModel(
        {
            "domain_sizes": domain_sizes,
            "grid_type": "cartesian",
            "meshing_arguments": {"cell_size": 5.0},
        }
    )
    model.set_geometry()
    mdg = model.mdg

    # Both fractures and their intersection should be represented in the grid.
    assert mdg.dim_max() == 3
    assert mdg.dim_min() == 1
    assert len(mdg.subdomains(dim=2)) == 2
    assert len(mdg.subdomains(dim=1)) == 1

    # The 3d grid should fill the domain, which extends downwards from the surface.
    sd = mdg.subdomains(dim=3)[0]
    assert np.allclose(sd.nodes.min(axis=1), [0.0, 0.0, -domain_sizes[2]])
    assert np.allclose(sd.nodes.max(axis=1), [domain_sizes[0], domain_sizes[1], 0.0])
    assert np.isclose(sd.cell_volumes.sum(), np.prod(domain_sizes))

    # The intersection line should be where the two fractures cross.
    intersection = mdg.subdomains(dim=1)[0]
    assert np.allclose(intersection.cell_centers[0], 5.0)
    assert np.allclose(intersection.cell_centers[2], -15.0)
