"""Minimal example for using AirflowEstimator without ScenarioConfig."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NS_wind_est import AirflowEstimator
from NS_wind_est import Visualizer

# Mesh file and measurement samples csv flile
MESH = ROOT / "example_cases/10x6_apartment/geometry/apartment_2d_coarse.msh"
SAMPLES = ROOT / "example_cases/10x6_apartment/samples/sample_points_n400_seed0.csv"

def main() -> None:

    # Create estimator
    estimator = AirflowEstimator.from_mesh(MESH)
    domain = estimator.domain

    # Set known boundary conditions
    estimator.set_no_slip_bc("Walls")
    estimator.set_zero_pressure_bc("Outlet")

    # Pass the measurements to the estimator
    mapping = estimator.set_measurements_from_csv(
        SAMPLES,
        count=50,
        noise_std=0.1,
        max_xy_dist=0.2,
    )

    # Configure solver parameters
    estimator.set_regularization("gradient")
    estimator.set_weights(
        viscosity=1.0e-5,
        weight_misfit=10.0,
        weight_pde_res=1.0,
        weight_reg=1.0e-2,
    )

    # Solve estimation problem
    solution = estimator.solve_SFNS(
        maxit=25,
        solver_tol=1.0e-2,
        damping=0.5,
        verbose=True,
    )

    # Show result
    vis = Visualizer(domain)
    vis.add_streamplot(solution.velocity)
    vis.add_background_mesh()
    vis.draw_boundary_segment(
        estimator.get_boundary_segments("Walls"),
        color="black",
        linewidth=2.0,
        label="Walls",
    )
    vis.draw_boundary_segment(
        estimator.get_boundary_segments(["Inlet_lower", "Inlet_left"]),
        color="orange",
        linewidth=3.0,
        label="Inflow"
    )
    vis.draw_boundary_segment(
        estimator.get_boundary_segments("Outlet"),
        color="red",
        linewidth=3.0,
        label="Outlet"
    )
    vis.show(title="Wind estimate")

    if domain.comm.rank == 0:
        print(f"Solver status: {solution.status}")

if __name__ == "__main__":
    main()
