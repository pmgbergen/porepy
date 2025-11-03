"""
This module contains the implementation of Case 2 from the 3D flow benchmark [1].

Note:
    The class `FlowBenchmark3dCase2Model` admits the parameter keyword
    `refinement_level`, which can take values 0, 1, 2, to control the mesh refinement
    level. Level `0` contains approximately 500 three-dimensional cells, level `1`
    contains 4K three-dimensional cells, and level `2` contains 32K three-dimensional
    cells.

    To set up case (a) with conductive fractures, use `solid_constants_conductive`.
    To set up case (b) with blocking fractures, use `solid_constants_blocking`.

References:
    [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
        E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
        three-dimensional fractured porous media. Advances in Water Resources, 147,
        103759.

"""