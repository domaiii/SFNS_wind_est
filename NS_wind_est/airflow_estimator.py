"""High-level API for Navier-Stokes based airflow estimation."""

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass

from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
import dolfinx.io as dio
from NS_wind_est.airflow_solvers import (
    AirflowSolverConfig,
    WfnsSolver,
    SfnsSolver
)

@dataclass(frozen=True, slots=True)
class SolverStatus:
    """Solver information reported during estimation."""

    solver: str
    converged: bool
    iterations: int
    max_iterations: int
    solver_tolerance: float
    final_relative_change: float


@dataclass(frozen=True, slots=True)
class AirflowResult:
    """Result returned by an airflow solve.

    Contains the collapsed velocity field, the collapsed pressure field,
    the original mixed velocity-pressure function, and solver status
    information.
    """

    velocity: fem.Function
    pressure: fem.Function
    mixed: fem.Function
    status: SolverStatus


class AirflowEstimator:
    """Estimate a wind field from sparse velocity measurements and an FE domain.

    The estimator exposes the direct Python API. It keeps all state that belongs
    to one estimation setup in one object: mesh, mixed function space, boundary
    conditions, measurements, solver configuration, and the `AirflowResult`.
    """

    def __init__(
        self,
        domain: mesh.Mesh,
        facet_tags: mesh.MeshTags,
        boundary_name_to_id: dict[str, int],
    ):
        """Create an estimator from an existing DOLFINx domain."""
        self.domain = domain
        self.facet_tags = facet_tags

        (
            self.W,
            self.W0,
            self.W1,
            self.V,
            self.Q,
            self.V_to_W,
            self.Q_to_W,
        ) = self.build_mixed_space(domain)

        self.w_measured = fem.Function(self.W)
        self.w_measured.x.array[:] = 0.0
        self.measurement_ids_W = np.array([], dtype=np.int32)

        self.domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )

        self.viscosity = 1.5e-5
        self.weight_misfit = 1e2
        self.weight_pde_res = 1e0
        self.weight_reg = 1e-2
        self.weight_wfns_bc = 1e0

        self.regularization_mode = "gradient"

        self.bcs: list[fem.DirichletBC] = []
        self.last_result: AirflowResult | None = None
        self._boundary_name_to_id: dict[str, int] = dict(boundary_name_to_id)

    @classmethod
    def from_mesh(
        cls,
        meshfile: str | Path,
        *,
        comm=MPI.COMM_WORLD
        ) -> "AirflowEstimator":
        """Create an estimator directly from a Gmsh `.msh` file.

        Physical boundary names from the Gmsh file are loaded as well, so
        named helpers such as :meth:`set_no_slip_bc` can be used immediately.
        """
        msh = Path(meshfile).resolve(strict=True)
        domain, _, facet_tags = dio.gmshio.read_from_msh(str(msh), comm, gdim=2)
        boundary_name_to_id = cls._read_physical_name_map(msh, dim=domain.topology.dim - 1)
        return cls(domain, facet_tags, boundary_name_to_id)

    @property
    def boundary_names(self) -> tuple[str, ...]:
        """Names of all available physical boundary groups."""
        return tuple(self._boundary_name_to_id.keys())

    @property
    def boundary_name_to_id(self) -> dict[str, int]:
        """Mapping from physical boundary names to mesh tag ids."""
        return dict(self._boundary_name_to_id)

    def match_boundary_names(self, pattern: str) -> list[str]:
        """Return physical boundary names matching a case-insensitive regex."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [name for name in self.boundary_names if regex.search(name)]

    def _boundary_ids(self, names: str | list[str]) -> list[int]:
        """Return boundary id(s) for given boundary name(s)."""
        if isinstance(names, str):
            names = [names]
        else:
            names = list(names)

        missing = [name for name in names if name not in self._boundary_name_to_id]
        if missing:
            available = ", ".join(self.boundary_names) or "<none>"
            raise ValueError(
                f"Unknown boundary name(s): {missing}. "
                f"Available boundaries: {available}"
            )

        return [self._boundary_name_to_id[name] for name in names]

    def get_boundary_segments(self, names: str | list[str]) -> list[np.ndarray]:
        """Return XY line segments for named physical boundary groups."""
        fdim = self.domain.topology.dim - 1
        self.domain.topology.create_connectivity(fdim, 0)
        facet_to_vertex = self.domain.topology.connectivity(fdim, 0)

        segments: list[np.ndarray] = []
        for boundary_id in self._boundary_ids(names):
            for facet in self.facet_tags.find(boundary_id):
                vertices = facet_to_vertex.links(int(facet))
                xy = self.domain.geometry.x[vertices, :2]
                if len(xy) == 0:
                    continue
                segments.append(np.asarray(xy, dtype=float))
        return segments

    def set_no_slip_bc(self, no_slip_bdry_names: str | list[str]):
        """
        Apply no-slip (u=0) boundary condition on the given physical group(s).
        """
        boundary_ids = self._boundary_ids(no_slip_bdry_names)
        facets = np.concatenate([
            self.facet_tags.find(boundary_id) for boundary_id in boundary_ids
        ])

        u_D = fem.Function(self.V)
        u_D.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W0, self.V),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(u_D, dofs, self.W0)
        self.add_dirichlet_bc(bc)
        return bc

    def set_zero_pressure_bc(self, pressure0_bdry_names: str | list[str]):
        """
        Apply p=0 boundary condition on the given outlet physical group(s).
        """
        boundary_ids = self._boundary_ids(pressure0_bdry_names)
        facets = np.concatenate([
            self.facet_tags.find(boundary_id) for boundary_id in boundary_ids
        ])

        p_zero = fem.Function(self.Q)
        p_zero.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W1, self.Q),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(p_zero, dofs, self.W1)
        self.add_dirichlet_bc(bc)
        return bc


    @staticmethod
    def build_mixed_space(domain, deg_u=2, deg_p=1):
        """Build the mixed velocity-pressure function spaces."""
        elem_u = element("Lagrange", domain.basix_cell(), deg_u, shape=(domain.geometry.dim,))
        elem_p = element("Lagrange", domain.basix_cell(), deg_p)
        mixed_elem = mixed_element([elem_u, elem_p])

        W = fem.functionspace(domain, mixed_elem)
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()
        return W, W0, W1, V, Q, np.array(V_to_W, dtype=np.int32), np.array(Q_to_W, dtype=np.int32)

    @staticmethod
    def _read_physical_name_map(
        meshfile: str | Path, dim: int | None = None
    ) -> dict[str, int]:
        """Read physical groups with name (str) and group id (int). """
        import gmsh

        meshfile = Path(meshfile).resolve(strict=True)
        gmsh.initialize()
        try:
            gmsh.open(str(meshfile))
            groups = gmsh.model.getPhysicalGroups(dim)
            return {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in groups}
        finally:
            gmsh.finalize()

    def set_regularization(self, mode: str):
        """Set the default regularization mode.

        Available modes are:

        - `"gradient"`: penalizes spatial variation of the velocity field via
          `||grad(u)||`. This favors smooth wind fields.
        - `"magnitude"`: penalizes the velocity magnitude via `||u||`. This
          favors smaller wind speeds where measurements and PDE terms allow it.
        """
        norm_mode = mode.strip().lower()
        if norm_mode not in {"gradient", "magnitude"}:
            raise ValueError("regularization mode must be 'gradient' or 'magnitude'.")

        self.regularization_mode = norm_mode

    def _build_solver_context(self) -> AirflowSolverConfig:
        """Configuration parameters for solver initialization."""
        return AirflowSolverConfig(
            domain=self.domain,
            W=self.W,
            V=self.V,
            V_to_W=self.V_to_W,
            bcs=self.bcs,
            w_measured=self.w_measured,
            measurement_ids_W=self.measurement_ids_W,
            viscosity=self.viscosity,
            weight_misfit=self.weight_misfit,
            weight_pde_res=self.weight_pde_res,
            weight_reg=self.weight_reg,
            weight_wfns_bc=self.weight_wfns_bc,
            regularization_mode=self.regularization_mode,
        )

    def _build_result(
        self, mixed: fem.Function, solver_status: dict, solver_name: str
    ) -> AirflowResult:
        """Returns `AirflowResult` after the estimation has terminated."""
        status = SolverStatus(
            solver=solver_name,
            converged=bool(solver_status["converged"]),
            iterations=int(solver_status["iterations"]),
            max_iterations=int(solver_status["max_iterations"]),
            solver_tolerance=float(solver_status["solver_tolerance"]),
            final_relative_change=float(solver_status["final_relative_change"]),
        )
        result = AirflowResult(
            velocity = mixed.sub(0).collapse(),
            pressure = mixed.sub(1).collapse(),
            mixed = mixed,
            status = status,
        )
        self.last_result = result
        return result

    def solve_SFNS(
        self,
        maxit: int = 25,
        solver_tol: float = 1e-2,
        damping: float | None = None,
        verbose: bool = False,
    ) -> AirflowResult:
        """Solve with the strong-form Navier-Stokes estimator.

        This solver minimizes the strong-form PDE residual, measurement
        mismatch, and regularization terms. Boundary conditions are imposed
        strongly through Dirichlet constraints.

        Parameters
        ----------
        maxit
            Maximum number of fixed-point iterations.
        solver_tol
            Relative change tolerance for fixed-point convergence.
        damping
            Optional damping factor for fixed-point updates. If `None`, no damping
            is applied.
        verbose
            If `True`, print iteration diagnostics.
        """
        solver = SfnsSolver(self._build_solver_context())
        mixed = solver.solve(
            maxit=maxit,
            solver_tol=solver_tol,
            damping=damping,
            verbose=verbose,
        )
        return self._build_result(mixed, solver.last_status, "SFNS")

    def solve_WFNS(
        self,
        maxit: int = 10,
        solver_tol: float = 1e-3,
        damping: float | None = None,
        verbose: bool = False,
    ) -> AirflowResult:
        """Solve with the weak-form Navier-Stokes estimator.

        This solver treats the weak-form PDE residual, measurement mismatch,
        regularization, and boundary conditions as weighted least-squares terms.
        It is kept available for comparison with the strong-form workflow.

        Parameters
        ----------
        maxit
            Maximum number of fixed-point iterations.
        solver_tol
            Relative change tolerance for fixed-point convergence.
        damping
            Optional damping factor for fixed-point updates. If `None`, no damping
            is applied.
        verbose
            If `True`, print iteration diagnostics.

        References
        ----------
        Wiedemann, T., Scheffler, M., Shutin, D., & Lilienthal, A. J. (2025).
        Physics-informed robotic airflow exploration and mapping with a
        swarm of mobile robots. The International Journal of Robotics
        Research, 44(13), 2105-2125.
        """

        solver = WfnsSolver(self._build_solver_context())
        mixed = solver.solve(
            maxit=maxit,
            solver_tol=solver_tol,
            damping=damping,
            verbose=verbose,
        )
        return self._build_result(mixed, solver.last_status, "WFNS")

    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        """Register one or more DOLFINx Dirichlet boundary conditions."""
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

    def set_measurements(
        self,
        measurement_ids_W: np.ndarray,
        measurement_values: np.ndarray,
        clear_existing: bool = True,
    ):
        """Set explicit wind measurements in mixed space W.

        Sets the measurements manually at flattend W-indices for velocity
        components. `measurement_values` are the flattened measurement values
        aligned with `measurement_ids_W`. If `clear_existing` set to `True`,
        all previously added/saved measurements are cleared first.
        """
        ids = np.asarray(measurement_ids_W, dtype=np.int32).reshape(-1)
        values = np.asarray(measurement_values, dtype=float).reshape(-1)

        if ids.size == 0:
            raise ValueError("measurement_ids_W is empty.")
        if ids.size != values.size:
            raise ValueError(
                f"Length mismatch: len(ids)={ids.size} != len(values)={values.size}"
            )
        if np.any(ids < 0) or np.any(ids >= self.w_measured.x.array.size):
            raise ValueError("measurement_ids_W contains out-of-bounds indices.")

        if clear_existing:
            self.w_measured.x.array[:] = 0.0

        self.w_measured.x.array[ids] = values
        self.measurement_ids_W = ids
        self.last_result = None

    def set_measurements_from_csv(
        self,
        samples_csv: str | Path,
        count: int | None = None,
        noise_std: float | None = None,
        max_xy_dist: float | None = None,
    ) -> dict[str, float]:
        """Load wind samples from CSV and map them to nearest velocity nodes.

        The CSV must contain `Points:0`, `Points:1`, `U:0` and `U:1`.
        Further columns, such as `Points:2` or `U:2`, are allowed but
        ignored. Samples are assigned to velocity nodes by nearest-neighbor
        mapping in XY coordinates.

        Parameters
        ----------
        samples_csv
            Path to the CSV file containing sparse wind measurements.
        count
            Optional number of rows to use only a part of the measurements in
            the CSV. If `None`, all are used.
        noise_std
            Optional standard deviation of zero-mean Gaussian noise added to
            the measured velocity components. If `None`, no noise is added.
        max_xy_dist
            Optional maximum allowed XY distance between a CSV sample point and
            the nearest velocity node. If exceeded, a `ValueError` is raised.

        Returns
        -------
        dict[str, float]
            Mapping metadata with the number of input samples, number of used
            samples, number of dropped duplicate-node samples, and maximum XY
            mapping distance.
        """
        samples_csv = Path(samples_csv).resolve(strict=True)
        df = pd.read_csv(samples_csv)

        if count is not None:
            count = int(count)
            if count < 1:
                raise ValueError(f"count must be at least 1, got {count}.")
            if count > len(df):
                raise ValueError(
                    f"Requested {count} measurements from {samples_csv.name}, "
                    f"but file only contains {len(df)} rows."
                )
            df = df.iloc[:count].copy()

        required = ["Points:0", "Points:1", "U:0", "U:1"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {samples_csv.name}: {missing}. "
                f"Expected at least {required}."
            )
        if len(df) == 0:
            raise ValueError(f"No sample rows found in {samples_csv}")

        samples_xy = df[["Points:0", "Points:1"]].to_numpy(dtype=float)
        samples_uv = df[["U:0", "U:1"]].to_numpy(dtype=float, copy=True)
        if noise_std is not None:
            samples_uv += np.random.normal(0, noise_std, (len(df), 2))

        node_xy = np.asarray(self.V.tabulate_dof_coordinates(), dtype=float)[:, :2]
        tree = cKDTree(node_xy)
        dist, node_ids = tree.query(samples_xy, k=1, p=2.0, workers=-1)

        max_dist = float(np.max(dist))
        if max_xy_dist is not None and max_dist > max_xy_dist:
            raise ValueError(
                f"Maximum XY mapping distance exceeded: "
                f"{max_dist:.6g} > {max_xy_dist:.6g}"
            )

        n_input = int(len(node_ids))
        _, first_idx = np.unique(node_ids, return_index=True)
        keep = np.sort(first_idx)
        n_dropped = n_input - int(len(keep))
        node_ids = node_ids[keep]
        samples_uv = samples_uv[keep]
        dist = dist[keep]

        dim = self.domain.geometry.dim
        velocity_ids_V = (
            node_ids[:, None] * dim + np.arange(dim, dtype=np.int32)
        ).reshape(-1)
        measurement_ids_W = self.V_to_W[velocity_ids_V]
        measurement_values = samples_uv[:, :dim].reshape(-1)

        self.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=True,
        )

        return {
            "n_input_samples": float(n_input),
            "n_used_samples": float(len(node_ids)),
            "n_dropped_duplicate_nodes": float(n_dropped),
            "max_xy_dist": float(np.max(dist)) if len(dist) else 0.0,
        }

    def set_weights(
        self,
        *,
        viscosity: float | None = None,
        weight_misfit: float | None = None,
        weight_pde_res: float | None = None,
        weight_reg: float | None = None,
        weight_wfns_bc: float | None = None,
    ) -> None:
        """Update weights of the solver objective terms.

        Arguments left as `None` keep their previous values.

        Parameters
        ----------
        viscosity
            Kinematic viscosity of the medium in m^2/s. The estimator default is
            1.5e-5, roughly the viscosity of air.
        weight_misfit
            Weights the mismatch between the measurements and the estimated solution
            at the corresponding locations.
        weight_pde_res
            Weights how well the solution satisfies the Navier-Stokes equations.
        weight_reg
            Weights the selected regularization term.
        weight_wfns_bc
            Weights the boundary-condition penalty used by WFNS. SFNS enforces
            boundary conditions strongly, so this value has no effect there.
        """
        if viscosity is not None:       self.viscosity = viscosity
        if weight_misfit is not None:   self.weight_misfit = weight_misfit
        if weight_pde_res is not None:  self.weight_pde_res = weight_pde_res
        if weight_reg is not None:      self.weight_reg = weight_reg
        if weight_wfns_bc is not None:  self.weight_wfns_bc = weight_wfns_bc

    def get_measurement_coordinates(self) -> np.ndarray:
        """Return coordinates of the currently configured measurements."""
        coords = self.V.tabulate_dof_coordinates()
        W_to_V = {w_id: v_id for v_id, w_id in enumerate(self.V_to_W)}
        measured_v_ids = np.asarray(
            [W_to_V[w_id] for w_id in self.measurement_ids_W if w_id in W_to_V],
            dtype=np.int32,
        )
        point_ids = np.unique(measured_v_ids // self.domain.geometry.dim)
        return coords[point_ids]
