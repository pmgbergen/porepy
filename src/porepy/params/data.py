""" Contains class for storing data / parameters associated with a grid.
"""
import warnings
import numpy as np

from porepy.numerics.mixed_dim.solver import Solver
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.bc import BoundaryCondition


class Parameters(object):
    """ Class to store all physical parameters used by solvers.

    The intention is to provide a unified way of passing around parameters, and
    also circumvent the issue of using a solver for multiple physical
    processes (e.g. different types of boundary conditions in multi-physics
    applications).

    List of possible attributes (implemented as properties, see respective
    getter methods for further description):

    General quantities (not physics specific)

    Scalar quantities
        biot_alpha: Biot's coefficient in poro-elasticity. Defaults to 1.
        fluid_viscosity: In a single phase system. Defaults to 1.
        fluid_compr: Fluid compressibility in a single phase system. Defaults
            to 0.

    Cell-wise quantities:
        aperture (fracture width). Defaults to 1.
        porosity

    Physic-specific
        tensor (Returns permeability, conductivtiy or stiffness)
        bc (BoundaryCondition object)
        bc_val (Boundary values
        sources (flow and transport)

    Solvers will access data as needed. If a solver inquires for unassigned
    data, this will result in a runtime error.

    Attributes (in addition to parameters described above):

    known_physics : list
        List of keyword signifying physical processes. There are at least one
        Solver that uses each of the keywords.

    """

    def __init__(self, g):
        """ Initialize Data object.

        Parameters:

        g - grid:
            Grid where the data is valid. Currently, only number of cells and
            faces are accessed.

        """
        self._num_cells = g.num_cells
        self._num_faces = g.num_faces
        self.dim = g.dim
        self.g = g

        self.known_physics = ["flow", "transport", "mechanics"]

    def __repr__(self):
        s = "Data object for grid with " + str(self._num_cells)
        s += " cells and " + str(self._num_faces) + " faces \n"
        s += "Assigned attributes / properties: \n"
        s += str(list(self.__dict__.keys()))
        return s

    def _get_physics(self, obj):
        if isinstance(obj, Solver):
            if not hasattr(obj, "physics"):
                raise AttributeError("Solver object should have attribute physics")
            s = obj.physics.strip().lower()
        elif isinstance(obj, str):
            s = obj.strip().lower()
        else:
            raise ValueError("Expected str or Solver object")

        if not s in self.known_physics:
            # Give a warning if the physics keyword is unknown
            warnings.warn("Unknown physics " + s)
        return s

    # ------------- Start of definition of parameters -------------------------

    # ------------- Constants

    # --------------- Biot Alpha ---------------------------------------
    def get_biot_alpha(self):
        if hasattr(self, "_biot_alpha"):
            return self._biot_alpha
        else:
            return 1

    def set_biot_alpha(self, val):
        if val < 0 or val > 1:
            raise ValueError("Biot's constant should be between 0 and 1")
        self._biot_alpha = val

    biot_alpha = property(get_biot_alpha, set_biot_alpha)

    # --------------- Fluid viscosity ------------------------------------
    def get_fluid_viscosity(self):
        if hasattr(self, "_fluid_viscosity"):
            return self._fluid_viscosity
        else:
            return 1

    def set_fluid_viscosity(self, val):
        if val <= 0:
            raise ValueError("Fluid viscosity should be positive")
        self._fluid_viscosity = val

    fluid_viscosity = property(get_fluid_viscosity, set_fluid_viscosity)

    # ----------- Fluid compressibility
    def get_fluid_compr(self):
        if hasattr(self, "_fluid_compr"):
            return self._fluid_compr
        else:
            return 0.

    def set_fluid_compr(self, val):
        if val < 0:
            raise ValueError("Fluid compressibility should be non-negative")
        self._fluid_compr = val

    fluid_compr = property(get_fluid_compr, set_fluid_compr)

    # ----------- Fluid density - treated as constant for now
    def get_fluid_density(self):
        if hasattr(self, "_fluid_density"):
            return self._fluid_density
        else:
            return 1

    def set_fluid_density(self, val):
        if val < 0:
            raise ValueError("Fluid density should be non-negative")
        self._fluid_density = val

    fluid_density = property(get_fluid_density, set_fluid_density)

    # ----------- Fluid specific heat - treated as constant for now
    def get_fluid_specific_heat(self):
        if hasattr(self, "_fluid_specific_heat"):
            return self._fluid_specific_heat
        else:
            return 1

    def set_fluid_specific_heat(self, val):
        if val < 0:
            raise ValueError("Fluid specific heat should be non-negative")
        self._fluid_specific_heat = val

    fluid_specific_heat = property(get_fluid_specific_heat, set_fluid_specific_heat)

    # ----------- rock density - treated as constant for now
    def get_rock_density(self):
        if hasattr(self, "_rock_density"):
            return self._rock_density
        else:
            return 1

    def set_rock_density(self, val):
        if val < 0:
            raise ValueError("rock density should be non-negative")
        self._rock_density = val

    rock_density = property(get_rock_density, set_rock_density)

    # ----------- rock specific heat - treated as constant for now
    def get_rock_specific_heat(self):
        if hasattr(self, "_rock_specific_heat"):
            return self._rock_specific_heat
        else:
            return 1

    def set_rock_specific_heat(self, val):
        if val < 0:
            raise ValueError("rock specific heat should be non-negative")
        self._rock_specific_heat = val

    rock_specific_heat = property(get_rock_specific_heat, set_rock_specific_heat)

    # -------------------- Cell-wise quantities below here --------------------

    # ------------------ Aperture -----------------

    def get_aperture(self, default=1):
        """ double or array_like
        Cell-wise quantity representing fracture aperture (really, height of
        surpressed dimensions). Set as either a np.ndarray, or a scalar
        (uniform) value. Always returned as np.ndarray.
        """
        if not hasattr(self, "_apertures"):
            return default * np.ones(self._num_cells)

        if isinstance(self._apertures, np.ndarray):
            # Hope that the user did not initialize as array with wrong size
            return self._apertures
        else:
            return self._apertures * np.ones(self._num_cells)

    def set_aperture(self, val):
        if np.any(val < 0):
            raise ValueError("Negative aperture")
        self._apertures = val

    aperture = property(get_aperture, set_aperture)

    # ---------------- Porosity -------------------------------------------------

    def get_porosity(self, default=1):
        """ double or array-like
        Cell-wise representation of porosity. Set as either a np.ndarary, or a
        scalar (uniform) value. Always returned as np.ndarray.
        """
        if not hasattr(self, "_porosity"):
            return default * np.ones(self._num_cells)

        if isinstance(self._porosity, np.ndarray):
            # Hope that the user did not initialize as array with wrong size
            return self._porosity
        else:
            return self._porosity * np.ones(self._num_cells)

    def set_porosity(self, val):
        if isinstance(val, np.ndarray):
            if np.any(val < 0) or np.any(val > 1):
                raise ValueError("Porosity outside unit interval")
        else:
            if val < 0 or val > 1:
                raise ValueError("Porosity outside unit interval")
        self._porosity = val

    porosity = property(get_porosity, set_porosity)

    # -------------------- Face-wise quantities below here --------------------

    # ------------------ slip_distance -----------------

    # ------------------------------------
    def get_slip_distance(self, default=0):
        """ double or array-like
        face-wise representation of slip distance. Set as either a np.ndarary, or a
        scalar (uniform) value. Always returned as np.ndarray.
        """
        if not hasattr(self, "_slip_distance"):
            return default * np.ones(self._num_faces * self.dim)

        if isinstance(self._slip_distance, np.ndarray):
            # Hope that the user did not initialize as array with wrong size
            return self._slip_distance
        else:
            return self._slip_distance * np.ones(self._num_faces * self.dim)

    def set_slip_distance(self, val):
        """ Set physics-specific slip_distance

        Parameters:

        val : slip distance, representing the difference in slip between left
              and right fracture faces
        """
        self._slip_distance = val

    slip_distance = property(get_slip_distance, set_slip_distance)

    # ----------- Multi-physics (solver-/context-dependent) parameters below -----

    # ------------------- Sources ---------------------------------------------

    def get_source(self, obj):
        """ Pick out physics-specific source.

        Discretization methods should access this method.

        Parameters:

        obj : Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        Returns:

        np.ndarray
            Volume source if obj.physics equals 'flow'
            Heat source if obj.physics equals 'transport'.

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            return self.get_source_flow()
        elif physics == "transport":
            return self.get_source_transport()
        elif physics == "mechanics":
            return self.get_source_mechanics()
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def set_source(self, obj, val):
        """ Set physics-specific source

        Parameters:

        obj: Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        val: np.ndarray. Size self._num_cells
            Source terms in each cell.

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            self._source_flow = val
        elif physics == "transport":
            self._source_transport = val
        elif physics == "mechanics":
            self._source_mechanics = val
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def get_source_flow(self):
        """ array_like
        Cell-wise quantity representing the volume source in a cell. Represent
        total in/outflow in the cell (integrated over the cell volume).
        Sources should be accessed via get_source / set_source
        """
        if hasattr(self, "_source_flow"):
            return self._source_flow
        else:
            return np.zeros(self._num_cells)

    source_flow = property(get_source_flow)

    def get_source_transport(self):
        """ array_like
        Cell-wise quantity representing the concentration / temperature source
        in a cell. Represent total in/outflow in the cell (integrated over the
        cell volume).
        Sources should be accessed via get_source / set_source
        """
        if hasattr(self, "_source_transport"):
            return self._source_transport
        else:
            return np.zeros(self._num_cells)

    source_transport = property(get_source_transport)

    def get_source_mechanics(self):
        if hasattr(self, "_source_mechanics"):
            return self._source_mechanics
        else:
            return np.zeros(self._num_cells * self.dim)

    source_mechanics = property(get_source_mechanics)

    # -------------------- Backgound stress, ---------------------
    def get_background_stress(self, obj):
        """ Pick out physics-specific background_stress.

        Discretization methods should access this method.

        Parameters:

        obj : Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        Returns:

        np.ndarray
            stress matrix size = (g.dim, g.dim)

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            return self.get_background_stress_flow()
        elif physics == "transport":
            return self.get_background_stress_transport()
        elif physics == "mechanics":
            return self.get_background_stress_mechanics()
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def set_background_stress(self, obj, val):
        """ Set physics-specific background_stress

        Parameters:

        obj: Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        val: np.ndarray. Size (g.dim, g.dim)
            background_stress for all cells

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            self._background_stress_flow = val
        elif physics == "transport":
            self._background_stress_transport = val
        elif physics == "mechanics":
            self._background_stress_mechanics = val
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def get_background_stress_flow(self):
        """ array_like
        Cell-wise quantity representing the volume background_stress in a cell. Represent
        total in/outflow in the cell (integrated over the cell volume).
        Background_Stresss should be accessed via get_background_stress / set_background_stress
        """
        if hasattr(self, "_background_stress_flow"):
            return self._background_stress_flow
        else:
            return np.zeros(self._num_cells)

    background_stress_flow = property(get_background_stress_flow)

    def get_background_stress_transport(self):
        """ array_like
        Cell-wise quantity representing the concentration / temperature background_stress
        in a cell. Represent total in/outflow in the cell (integrated over the
        cell volume).
        Background_Stresss should be accessed via get_background_stress / set_background_stress
        """
        if hasattr(self, "_background_stress_transport"):
            return self._background_stress_transport
        else:
            return np.zeros(self._num_cells)

    background_stress_transport = property(get_background_stress_transport)

    def get_background_stress_mechanics(self):
        if hasattr(self, "_background_stress_mechanics"):
            return self._background_stress_mechanics
        else:
            return np.zeros(self._num_cells * self.dim)

    background_stress_mechanics = property(get_background_stress_mechanics)

    # -------------------- Permeability, conductivity, ---------------------

    def get_tensor(self, obj):
        """ Pick out physics-specific tensor.

        Discretization methods considering second and fourth orrder tensors
        (e.g. permeability, conductivity, stiffness) should access this method.

        Parameters:

        obj : Solver
            Discretization object. Should have attribute 'physics'.

        Returns:

        tensor, representing
            Permeability if obj.physics equals 'flow'
            conductivity if obj.physics equals 'transport'
            stiffness if physics equals 'mechanics'

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            return self.get_permeability()
        elif physics == "transport":
            return self.get_conductivity()
        elif physics == "mechanics":
            return self.get_stiffness()
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def set_tensor(self, obj, val):
        """ Set physics-specific tensor

        Parameters:

        obj: Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        val : tensor, representing
            Permeability if obj.physics equals 'flow'
            conductivity if obj.physics equals 'transport'
            stiffness if physics equals 'mechanics'

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            self._perm = val
        elif physics == "transport":
            self._conductivity = val
        elif physics == "mechanics":
            self._stiffness = val
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def get_permeability(self):
        """ tensor.SecondOrderTensor
        Cell wise permeability, represented as a second order tensor.
        Defaults to a unit tensor.
        """
        if hasattr(self, "_perm"):
            return self._perm
        else:
            t = SecondOrderTensor(self.dim, np.ones(self._num_cells))
            return t

    perm = property(get_permeability)

    def get_conductivity(self):
        """ tensor.SecondOrderTensor
        Cell wise conductivity, represented as a second order tensor.
        Defaults to a unit tensor.
        """
        if hasattr(self, "_conductivity"):
            return self._conductivity
        else:
            t = SecondOrderTensor(self.dim, np.ones(self._num_cells))
            return t

    conductivity = property(get_conductivity)

    def get_stiffness(self):
        """ Stiffness matrix, defined as fourth order tensor.
        If not defined, a unit tensor is returned.
        """
        if hasattr(self, "_stiffness"):
            return self._stiffness
        else:
            t = FourthOrderTensor(
                self.dim, np.ones(self._num_cells), np.ones(self._num_cells)
            )
            return t

    stiffness = property(get_stiffness)

    # --------------------- Boundary conditions and values ------------------------

    # Boundary condition

    def get_bc(self, obj):
        """ Pick out physics-specific boundary condition

        Discretization methods should access this method.

        Parameters:

        obj : Solver
            Discretization object. Should have attribute 'physics'.

        Returns:

        BoundaryCondition, for
            flow/pressure equation y if physics equals 'flow'
            transport equation if physics equals 'transport'
            elasticity if physics equals 'mechanics'

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            return self.get_bc_flow()
        elif physics == "transport":
            return self.get_bc_transport()
        elif physics == "mechanics":
            return self.get_bc_mechanics()
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def set_bc(self, obj, val):
        """ Set physics-specific boundary condition

        Parameters:

        obj: Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        val : BoundaryCondition, representing
            flow/pressure equation y if physics equals 'flow'
            transport equation if physics equals 'transport'
            elasticity if physics equals 'mechanics'

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            self._bc_flow = val
        elif physics == "transport":
            self._bc_transport = val
        elif physics == "mechanics":
            self._bc_mechanics = val
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def get_bc_flow(self):
        """ BoundaryCondition object
        Cell wise permeability, represented as a second order tensor.
        Solvers should rather access get_tensor().
        """
        if hasattr(self, "_bc_flow"):
            return self._bc_flow
        else:
            return BoundaryCondition(self.g)

    bc_flow = property(get_bc_flow)

    def get_bc_transport(self):
        """ bc.BoundaryCondition
        Cell wise conductivity, represented as a second order tensor.
        Solvers should rather access tensor().
        """
        if hasattr(self, "_bc_transport"):
            return self._bc_transport
        else:
            return BoundaryCondition(self.g)

    bc_transport = property(get_bc_transport)

    def get_bc_mechanics(self):
        """ Stiffness matrix, defined as fourth order tensor
        """
        if hasattr(self, "_bc_mechanics"):
            return self._bc_mechanics
        else:
            return BoundaryCondition(self.g)

    bc_mechanics = property(get_bc_mechanics)

    # Boundary value

    def get_bc_val(self, obj):
        """ Pick out physics-specific boundary condition

        Discretization methods should access this method.

        Parameters:

        obj : Solver
            Discretization object. Should have attribute 'physics'.

        Returns:

        BoundaryCondition, for
            flow/pressure equation y if physics equals 'flow'
            transport equation if physics equals 'transport'
            elasticity if physics equals 'mechanics'
            If the BoundaryCondition is not specified, Neumann conditions will
            be assigned.
        """
        physics = self._get_physics(obj)

        if physics == "flow":
            return self.get_bc_val_flow()
        elif physics == "transport":
            return self.get_bc_val_transport()
        elif physics == "mechanics":
            return self.get_bc_val_mechanics()
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def set_bc_val(self, obj, val):
        """ Set physics-specific boundary condition

        Parameters:

        obj: Solver or str
            Identification of physical regime. Either discretization object
            with attribute 'physics' or a str.

        val : np.array: Value for boundary condition, as specified by the
            relevant numerical method.

        """
        physics = self._get_physics(obj)

        if physics == "flow":
            self._bc_val_flow = val
        elif physics == "transport":
            self._bc_val_transport = val
        elif physics == "mechanics":
            self._bc_val_mechanics = val
        else:
            raise ValueError(
                'Unknown physics "%s".\n Possible physics are: %s'
                % (physics, self.known_physics)
            )

    def get_bc_val_flow(self):
        """ tensor.SecondOrderTensor
        Cell wise permeability, represented as a second order tensor.
        Solvers should rather access get_tensor().
        """
        if hasattr(self, "_bc_val_flow"):
            return self._bc_val_flow
        else:
            return np.zeros(self._num_faces)

    bc_val_flow = property(get_bc_val_flow)

    def get_bc_val_transport(self):
        """ tensor.SecondOrderTensor
        Cell wise conductivity, represented as a second order tensor.
        Solvers should rather access tensor().
        """
        if hasattr(self, "_bc_val_transport"):
            return self._bc_val_transport
        else:
            return np.zeros(self._num_faces)

    bc_val_transport = property(get_bc_val_transport)

    def get_bc_val_mechanics(self):
        """ tensor.FourthOrderTensor
        Cell wise conductivity, represented as a fourth order tensor.
        Solvers should rather access tensor().
        """
        if hasattr(self, "_bc_val_mechanics"):
            return self._bc_val_mechanics
        else:
            return np.zeros(self._num_faces * self.dim)

    bc_val_mechanics = property(get_bc_val_mechanics)
