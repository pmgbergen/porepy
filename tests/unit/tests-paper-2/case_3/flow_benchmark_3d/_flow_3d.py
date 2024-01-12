from pathlib import Path
import sys

if sys.version_info.minor < 8:
    from typing_extensions import Literal, Union, Optional
else:
    from typing import Literal, Union, Optional

import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d

def case1(
    refinement: Optional[Literal[0, 1, 2]] = None, only_network: Optional[bool] = False
) -> Union[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Case 2 in 3d flow benchmark.

    Parameters:
        refinement (int): Refinement level. Should be 0, 1 or 2,
            corresponding to grids with roughly 500, 4k and 32k
            3d cells, respectively.

    Returns:
        pp.MixedDimensionalGrid: Mixed-dimensional grid of the domain.

    """
    abs_path = Path(__file__)
    directory = abs_path.parent / Path("case1")

    if only_network:
        return pp.fracture_importer.network_3d_from_csv(
            str(directory / Path("benchmark_3d_case_1.csv"))
        )

    if refinement == 0:
        file_name = directory / Path("mesh1k.geo")
    elif refinement == 1:
        file_name = directory / Path("mesh10k.geo")
    elif refinement == 2:
        file_name = directory / Path("mesh100k.geo")
    else:
        raise ValueError(f"Expected refinement level 0, 1, or 2, got {refinement}")

    return pp.fracture_importer.dfm_from_gmsh(str(file_name), 3)


def case2(
    refinement: Optional[Literal[0, 1, 2]] = None, only_network: Optional[bool] = False
) -> Union[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Case 2 in 3d flow benchmark.

    Parameters:
        refinement (int): Refinement level. Should be 0, 1 or 2,
            corresponding to grids with roughly 500, 4k and 32k
            3d cells, respectively.

    Returns:
        pp.MixedDimensionalGrid: Mixed-dimensional grid of the domain.

    """
    abs_path = Path(__file__)
    directory = abs_path.parent / Path("case2")

    if only_network:
        return pp.fracture_importer.network_3d_from_csv(
            str(directory / Path("benchmark_3d_case_2.csv"))
        )

    if refinement == 0:
        file_name = directory / Path("mesh500.geo")
    elif refinement == 1:
        file_name = directory / Path("mesh4k.geo")
    elif refinement == 2:
        file_name = directory / Path("mesh32k.geo")
    else:
        raise ValueError(f"Expected refinement level 0, 1, or 2, got {refinement}")

    return pp.fracture_importer.dfm_from_gmsh(str(file_name), 3)


def case3(
    refinement: Optional[Literal[0, 1, 2, 3]] = None,
    only_network: Optional[bool] = False,
) -> Union[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Case 3 in 3d flow benchmark.

    Parameters:
        refinement (int): Refinement level. Should be 0, 1, 2 or 3.
            The first two levels (0 and 1) corresponds to roughly 30K and 140K
            3d cells as prescribed in the benchmark. The two latter refinements
            produce grids with 350K and 500K 3d cells (depending a bit on Gmsh
            version).

    Returns:
        pp.MixedDimensionalGrid: Mixed-dimensional grid of the domain.

    """
    abs_path = Path(__file__)
    directory = abs_path.parent / Path("case3")

    if only_network:
        return pp.fracture_importer.network_3d_from_csv(
            str(directory / Path("benchmark_3d_case_3.csv"))
        )

    if refinement == 0:
        file_name = directory / Path("mesh30k.geo")
    elif refinement == 1:
        file_name = directory / Path("mesh140k.geo")
    elif refinement == 2:
        file_name = directory / Path("mesh350k.geo")
    elif refinement == 3:
        file_name = directory / Path("mesh500k.geo")
    else:
        raise ValueError(f"Expected refinement level 0, 1, 2 or 3, got {refinement}")

    return pp.fracture_importer.dfm_from_gmsh(str(file_name), 3)


def case4(
    only_network: Optional[bool] = False,
) -> Union[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """Case 4 in 3d flow benchmark.

    For now, mesh size cannot be adjusted, only the grid as specified in the
    benchmark (modulu changes to gmsh) is available

    Returns:
        pp.MixedDimensionalGrid: Mixed-dimensional grid of the domain

    """
    abs_path = Path(__file__)
    directory = abs_path.parent / Path("case4")

    if only_network:
        return pp.fracture_importer.network_3d_from_csv(
            str(directory / Path("benchmark_3d_case_4.csv")), has_domain=False
        )

    file_name = directory / Path("mesh242k.geo")

    return pp.fracture_importer.dfm_from_gmsh(str(file_name), 3)
