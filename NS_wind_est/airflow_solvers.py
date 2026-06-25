"""Backend solvers for Navier-Stokes airflow estimation."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sps
import ufl

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from scipy.sparse.linalg import lsqr
from ufl import div, dot, dx, grad, inner


@dataclass(slots=True)
class AirflowSolverConfig:
    """Shared numerical context passed from the estimator to solver backends."""

    domain: mesh.Mesh
    W: fem.FunctionSpace
    V: fem.FunctionSpace
    V_to_W: np.ndarray
    bcs: list[fem.DirichletBC]
    w_measured: fem.Function
    measurement_ids_W: np.ndarray
    viscosity: float
    weight_misfit: float
    weight_pde_res: float
    weight_reg: float
    weight_wfns_bc: float
    regularization_mode: str


class BaseAirflowSolver(ABC):
    """Base class for fixed-point airflow solver backends."""

    def __init__(self, ctx: AirflowSolverConfig):
        self.ctx = ctx
        self.last_status = {
            "converged": False,
            "iterations": 0,
            "max_iterations": 0,
            "solver_tolerance": float("nan"),
            "final_relative_change": float("nan"),
        }

    def solve(self,
              maxit: int,
              solver_tol: float,
              damping: float | None = None,
              verbose: bool = False):
        """Run fixed-point iterations and return the mixed solution function."""
        wh = fem.Function(self.ctx.W)
        wh_prev = fem.Function(self.ctx.W)
        wh_prev.x.array[:] = 0.0

        reg_mode = self.ctx.regularization_mode
        self._validate()
        context = self._prepare_solve(reg_mode)

        iterations = 0
        converged = False
        final_diff = float("nan")

        for k in range(maxit):
            self._solve_step(wh_prev, wh, reg_mode, context)

            diff = np.linalg.norm(wh.x.array - wh_prev.x.array) / (np.linalg.norm(wh.x.array) + 1e-10)
            iterations = k + 1
            final_diff = float(diff)
            if diff < solver_tol:
                converged = True
                break

            if damping is not None and 0.0 < damping < 1.0:
                wh_prev.x.array[:] = (1 - damping) * wh_prev.x.array + damping * wh.x.array
            else:
                wh_prev.x.array[:] = wh.x.array

        self.last_status = {
            "converged": converged,
            "iterations": iterations,
            "max_iterations": int(maxit),
            "solver_tolerance": float(solver_tol),
            "final_relative_change": final_diff,
        }

        if verbose:
            self._report_summary(wh, reg_mode, context)
        self._finalize_solve(context)
        return wh

    def _validate(self):
        """Check solver prerequisites before iteration starts."""
        if not self.ctx.bcs:
            raise ValueError("No boundary conditions set. Use add_dirichlet_bc() to add BCs.")

    def _prepare_solve(self, reg_mode: str):
        """Build optional solver-specific data before the iteration loop."""
        return None

    def _finalize_solve(self, context):
        """Release optional solver-specific data after the iteration loop."""
        return None

    @staticmethod
    def _num_dofs(space) -> int:
        """Return the global number of scalar degrees of freedom."""
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs

    def _build_weak_form_system(self, wh_prev: fem.Function):
        """Assemble the linearized weak-form Navier-Stokes system."""
        (u, p) = ufl.TrialFunctions(self.ctx.W)
        (v, q) = ufl.TestFunctions(self.ctx.W)
        uh_prev, _ = ufl.split(wh_prev)

        nu = fem.Constant(self.ctx.domain, PETSc.ScalarType(self.ctx.viscosity))
        zero_vec = fem.Constant(
            self.ctx.domain,
            PETSc.ScalarType((0.0,) * self.ctx.domain.geometry.dim),
        )

        a_form = fem.form((
            inner(nu * grad(u), grad(v))
            + inner(grad(u) * uh_prev, v)
            - p * div(v)
            + q * div(u)
        ) * dx)
        f_form = fem.form(inner(zero_vec, v) * dx)

        K = assemble_matrix(a_form)
        K.assemble()
        f = assemble_vector(f_form)
        return K, f

    def _measurement_misfit(self, wh: fem.Function) -> float:
        """Evaluate squared mismatch at configured measurement dofs."""
        diff = wh.x.array - self.ctx.w_measured.x.array
        m_idx = self.ctx.measurement_ids_W
        return float(np.sum(diff[m_idx] ** 2))

    def _boundary_penalty(self, wh: fem.Function) -> float:
        """Evaluate squared values on dofs constrained by boundary conditions."""
        bc_dofs = []
        for bc in self.ctx.bcs:
            bc_dofs.extend(map(int, bc.dof_indices()[0]))
        if not bc_dofs:
            return 0.0
        values = wh.x.array[np.asarray(bc_dofs, dtype=np.int32)]
        return float(np.sum(values ** 2))

    def _magnitude_regularization(self, wh: fem.Function, reg_mode: str) -> float:
        """Evaluate the selected velocity regularization functional."""
        uh, _ = ufl.split(wh)
        if reg_mode == "magnitude":
            form = fem.form(inner(uh, uh) * dx)
        else:
            form = fem.form(inner(grad(uh), grad(uh)) * dx)
        return float(self.ctx.domain.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM))

    @abstractmethod
    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        """Compute one fixed-point update into `wh`."""
        pass

    def _report_summary(self, wh: fem.Function, reg_mode: str, context):
        """Print objective-term diagnostics for the final iterate."""
        terms = self._evaluate_terms(wh, reg_mode, context)
        if not terms:
            return
        print("Objective terms:")
        for key, value in terms.items():
            print(f"{key:>26}: {value:.6e}")

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        """Return objective-term diagnostics for reporting."""
        return {}


class SfnsSolver(BaseAirflowSolver):
    """Strong-form Navier-Stokes residual solver backend."""

    def _build_system(self, wh_prev: fem.Function, reg_mode: str):
        """Assemble the SFNS normal-equation system for one update."""
        W = wh_prev.function_space
        domain = W.mesh

        nu = fem.Constant(domain, PETSc.ScalarType(self.ctx.viscosity))
        beta = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_pde_res))
        gamma = fem.Constant(domain, PETSc.ScalarType(self.ctx.weight_reg))

        uh_prev, _ = wh_prev.split()
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)

        Rmom_u = -nu * div(grad(u)) + dot(uh_prev, grad(u)) + grad(p)
        Rmom_v = -nu * div(grad(v)) + dot(uh_prev, grad(v)) + grad(q)
        Rdiv_u = div(u)
        Rdiv_v = div(v)

        a_pde = (beta * (inner(Rmom_u, Rmom_v) + Rdiv_u * Rdiv_v)) * dx
        if reg_mode == "magnitude":
            a_reg = (gamma * inner(u, v)) * dx
        else:
            a_reg = (gamma * inner(grad(u), grad(v))) * dx

        zero_vec = fem.Constant(domain, PETSc.ScalarType((0.0,) * domain.geometry.dim))
        L = inner(zero_vec, v) * dx

        aF, LF = fem.form(a_pde + a_reg), fem.form(L)
        A = assemble_matrix(aF, bcs=self.ctx.bcs)
        A.assemble()
        b = assemble_vector(LF)
        fem.apply_lifting(b, [aF], bcs=[self.ctx.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, self.ctx.bcs)

        S = PETSc.Mat().createAIJ(A.getSizes(), nnz=1, comm=A.comm)
        S.setUp()
        for i in map(int, self.ctx.measurement_ids_W):
            S.setValue(i, i, 1.0)
        S.assemble()

        rhs_add = self.ctx.w_measured.x.petsc_vec.duplicate()
        S.mult(self.ctx.w_measured.x.petsc_vec, rhs_add)

        A.axpy(
            self.ctx.weight_misfit,
            S,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        b.axpy(self.ctx.weight_misfit, rhs_add)
        return A, b

    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        """Solve one SFNS linear system with PETSc LU."""
        A, b = self._build_system(wh_prev, reg_mode)
        ksp = PETSc.KSP().create(A.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.petsc_vec.assemblyBegin()
        wh.x.petsc_vec.assemblyEnd()
        wh.x.array[:] = wh.x.petsc_vec.getArray(readonly=True)

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        """Evaluate unweighted and weighted SFNS objective terms."""
        domain = self.ctx.domain
        nu = fem.Constant(domain, PETSc.ScalarType(self.ctx.viscosity))

        uh, ph = ufl.split(wh)
        Rmom = -nu * div(grad(uh)) + dot(uh, grad(uh)) + grad(ph)
        Rdiv = div(uh)

        pde_form = fem.form((inner(Rmom, Rmom) + Rdiv * Rdiv) * dx)
        pde = float(domain.comm.allreduce(fem.assemble_scalar(pde_form), op=MPI.SUM))
        reg = self._magnitude_regularization(wh, reg_mode)
        misfit = self._measurement_misfit(wh)

        return {
            "pde_unweighted": pde,
            "reg_unweighted": reg,
            "misfit_unweighted": misfit,
            "pde_weighted": self.ctx.weight_pde_res * pde,
            "reg_weighted": self.ctx.weight_reg * reg,
            "misfit_weighted": self.ctx.weight_misfit * misfit,
            "objective_total_weighted": (
                self.ctx.weight_pde_res * pde
                + self.ctx.weight_reg * reg
                + self.ctx.weight_misfit * misfit
            ),
        }

class WfnsSolver(BaseAirflowSolver):
    """Weak-form least-squares Navier-Stokes solver backend."""

    def __init__(self, ctx: AirflowSolverConfig):
        super().__init__(ctx)
        self._gradient_regularization_operator: sps.csr_matrix | None = None

    def _collect_bc_dofs(self) -> np.ndarray:
        """Collect scalar dofs touched by configured Dirichlet conditions."""
        bc_dofs: list[int] = []
        for bc in self.ctx.bcs:
            bc_dofs.extend(map(int, bc.dof_indices()[0]))
        return np.asarray(sorted(set(bc_dofs)), dtype=np.int32)

    @staticmethod
    def _selection_matrix(indices: np.ndarray, num_cols: int) -> sps.csr_matrix:
        """Build a sparse matrix selecting the requested scalar dofs."""
        indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        rows = np.arange(indices.size, dtype=np.int32)
        vals = np.ones(indices.size, dtype=float)
        return sps.csr_matrix((vals, (rows, indices)), shape=(indices.size, num_cols))

    def _build_linear_regularization_operator(self, reg_mode: str) -> sps.csr_matrix:
        """Build the sparse WFNS regularization operator."""
        num_total_dofs = self._num_dofs(self.ctx.W)

        if reg_mode == "magnitude":
            return sps.identity(num_total_dofs, format="csr")

        if self._gradient_regularization_operator is not None:
            return self._gradient_regularization_operator

        coords = np.asarray(
            self.ctx.V.tabulate_dof_coordinates(),
            dtype=float,
        )[:, :self.ctx.domain.geometry.dim]

        num_points = len(coords)
        if num_points < 2:
            self._gradient_regularization_operator = sps.csr_matrix((0, num_total_dofs))
            return self._gradient_regularization_operator

        k = min(5, num_points)
        tree = cKDTree(coords)
        dists, neighbors = tree.query(coords, k=k, p=2.0, workers=-1)

        rows, cols, vals = [], [], []
        row_id = 0
        seen_edges: set[tuple[int, int]] = set()
        dim = self.ctx.domain.geometry.dim

        for i in range(num_points):
            for dist_ij, j in zip(np.atleast_1d(dists[i])[1:], np.atleast_1d(neighbors[i])[1:]):
                j = int(j)
                edge = (i, j) if i < j else (j, i)
                if edge[0] == edge[1] or edge in seen_edges:
                    continue

                seen_edges.add(edge)
                weight = 1.0 / max(float(dist_ij), 1e-12)

                for comp in range(dim):
                    wi = int(self.ctx.V_to_W[i * dim + comp])
                    wj = int(self.ctx.V_to_W[j * dim + comp])
                    rows.extend([row_id, row_id])
                    cols.extend([wi, wj])
                    vals.extend([weight, -weight])
                    row_id += 1

        self._gradient_regularization_operator = sps.csr_matrix(
            (vals, (rows, cols)),
            shape=(row_id, num_total_dofs),
        )
        return self._gradient_regularization_operator

    def _prepare_solve(self, reg_mode: str):
        """Precompute sparse WFNS matrices that do not change per iteration."""
        num_total_dofs = self._num_dofs(self.ctx.W)

        bc_dofs = self._collect_bc_dofs()
        R_sp = self._selection_matrix(bc_dofs, num_total_dofs)

        m_idx = np.asarray(self.ctx.measurement_ids_W, dtype=np.int32).reshape(-1)
        M_sp = self._selection_matrix(m_idx, num_total_dofs)

        reg_op = self._build_linear_regularization_operator(reg_mode)

        free_pde_rows = np.ones(num_total_dofs, dtype=bool)
        free_pde_rows[bc_dofs] = False

        fixed_matrix = sps.vstack([
            np.sqrt(self.ctx.weight_wfns_bc) * R_sp,
            np.sqrt(self.ctx.weight_misfit) * M_sp,
            np.sqrt(self.ctx.weight_reg) * reg_op,
        ]).tocsr()

        fixed_rhs = np.concatenate([
            np.zeros(R_sp.shape[0]),
            np.sqrt(self.ctx.weight_misfit) * self.ctx.w_measured.x.array[m_idx],
            np.zeros(reg_op.shape[0]),
        ])

        return {
            "num_total_dofs": num_total_dofs,
            "sqrt_w_pde": np.sqrt(self.ctx.weight_pde_res),
            "free_pde_rows": free_pde_rows,
            "reg_op": reg_op,
            "fixed_matrix": fixed_matrix,
            "fixed_rhs": fixed_rhs,
        }

    def _solve_step(self, wh_prev: fem.Function, wh: fem.Function, reg_mode: str, context):
        """Solve one WFNS least-squares update with SciPy LSQR."""
        K_petsc, f_petsc = self._build_weak_form_system(wh_prev)

        ai, aj, av = K_petsc.getValuesCSR()
        K_sp = sps.csr_matrix(
            (av, aj, ai),
            shape=(context["num_total_dofs"], context["num_total_dofs"]),
        )

        free_rows = context["free_pde_rows"]
        K_sp = K_sp[free_rows, :]
        f_vec = np.asarray(f_petsc.array, dtype=float)[free_rows]

        A_stack = sps.vstack([
            context["sqrt_w_pde"] * K_sp,
            context["fixed_matrix"],
        ]).tocsr()

        b_stack = np.concatenate([
            -context["sqrt_w_pde"] * f_vec,
            context["fixed_rhs"],
        ])

        result = lsqr(A_stack, b_stack, iter_lim=5000)[0]
        wh.x.array[:] = result
        wh.x.scatter_forward()

    def _evaluate_terms(self, wh: fem.Function, reg_mode: str, context) -> dict[str, float]:
        """Evaluate unweighted and weighted WFNS objective terms."""
        K_petsc, f_petsc = self._build_weak_form_system(wh)

        residual = f_petsc.duplicate()
        K_petsc.mult(wh.x.petsc_vec, residual)
        residual.axpy(-1.0, f_petsc)

        residual_arr = np.asarray(residual.array, dtype=float)
        pde_residual = residual_arr[context["free_pde_rows"]]
        pde = float(np.dot(pde_residual, pde_residual))

        boundary = self._boundary_penalty(wh)
        misfit = self._measurement_misfit(wh)

        reg_vec = context["reg_op"] @ wh.x.array
        reg = float(np.dot(reg_vec, reg_vec))

        return {
            "pde_unweighted": pde,
            "reg_unweighted": reg,
            "boundary_unweighted": boundary,
            "misfit_unweighted": misfit,
            "pde_weighted": self.ctx.weight_pde_res * pde,
            "reg_weighted": self.ctx.weight_reg * reg,
            "boundary_weighted": self.ctx.weight_wfns_bc * boundary,
            "misfit_weighted": self.ctx.weight_misfit * misfit,
            "objective_total_weighted": (
                self.ctx.weight_pde_res * pde
                + self.ctx.weight_reg * reg
                + self.ctx.weight_wfns_bc * boundary
                + self.ctx.weight_misfit * misfit
            ),
        }
