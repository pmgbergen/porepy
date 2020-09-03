"""
Module indended for combining fracture propagation with complex multiphysics,
as represented in the model classes.

This module contains a partial implementation of propagation, confer other modules
for full implementation of the propagation.

WARNING: This should be considered experimental code and should be used with
    extreme caution. In particular, the code likely contains bugs, possibly of a
    severe character. Moreover, simulation of fracture propagation may cause
    numerical stability issues that it will likely take case-specific adaptations
    to resolve.



"""
import abc

import scipy.sparse as sps
import porepy as pp


class FracturePropagation(abc.ABC):
    """ Abstract base class for fracture propagation methods.

    The class is indended used together with a subclass of AbstractModel,
    using dual inheritance.

    Known subclasses are:
        ConformingFracturePropagation

    """

    @abc.abstractmethod
    def evaluate_propagation(self) -> None:
        """ Evaluate propagation of fractures based on the current solution.

        Impementation of the method will differ between propagation criretia,
        whether the adaptive meshing is applied etc.
        """

    @abc.abstractmethod
    def has_propagated(self) -> bool:
        """ Should return True if fractures were propagated in the previous step.
        """

    def update_variables(self):
        for g, d in self.gb:
            if not ("cell_index_map" in d and "face_index_map" in d):
                continue

            cell_map = d["cell_index_map"]
            face_map = d["face_index_map"]

            for var, dofs in d[pp.PRIMARY_VARIABLES].items():

                # Only cell-based dofs have been considered so far.
                # It should not be difficult to handle other types of variables,
                # but the need has not been there.
                face_dof: int = dofs.get("faces", 0)
                node_dof: int = dofs.get("nodes", 0)
                if face_dof != 0 or node_dof != 0:
                    raise NotImplementedError(
                        "Have only implemented variable mapping for face dofs"
                    )

                cell_dof: int = dofs.get("cells")

                mapping = sps.kron(cell_map, sps.eye(cell_dof))

                d[pp.STATE][var] = mapping * d[pp.STATE][var]
                if var in d[pp.STATE][pp.ITERATE].keys():
                    d[pp.STATE][pp.ITERATE][var] = (
                        mapping * d[pp.STATE][pp.ITERATE][var]
                    )

        for _, d in self.gb.edges():
            if not "cell_index_map" in d:
                continue

            cell_map = d["cell_index_map"]

            for var, dofs in d[pp.PRIMARY_VARIABLES].items():
                # Only cell-based dofs have been considered so far.
                cell_dof: int = dofs.get("cells")

                mapping = sps.kron(cell_map, sps.eye(cell_dof))

                d[pp.STATE][var] = mapping * d[pp.STATE][var]
                if var in d[pp.STATE][pp.ITERATE].keys():
                    d[pp.STATE][pp.ITERATE][var] = (
                        mapping * d[pp.STATE][pp.ITERATE][var]
                    )
        # Also update the assembler's counting of dofs
        self.assembler.update_dof_count()
