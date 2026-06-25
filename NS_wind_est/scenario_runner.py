"""Scenario workflow for configured airflow estimation cases.

This module contains the Python entry point behind the command-line scenario
runner. It loads a `ScenarioConfig`, configures an `AirflowEstimator`, runs one
or more sample CSVs, and writes result files plus metadata.
"""

import json
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dolfinx import fem

from NS_wind_est.airflow_estimator import AirflowEstimator, AirflowResult
from NS_wind_est.scenario import ScenarioConfig


@dataclass(frozen=True)
class ScenarioRunResult:
    """Paths produced by a scenario run."""

    output_root: Path
    result_dirs: tuple[Path, ...]

    @property
    def n_runs(self) -> int:
        """Number of individual result directories created by the run."""
        return len(self.result_dirs)


def save_velocity_csv(path: Path, velocity: fem.Function) -> None:
    """Write a velocity function to the CSV format used by the examples."""
    coords = velocity.function_space.tabulate_dof_coordinates()
    values = velocity.x.array.reshape(-1, velocity.function_space.dofmap.bs)

    z = coords[:, 2] if coords.shape[1] > 2 else np.zeros(coords.shape[0])
    w = values[:, 2] if values.shape[1] > 2 else np.zeros(values.shape[0])
    data = np.column_stack([coords[:, 0], coords[:, 1], z, values[:, 0], values[:, 1], w])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="Points:0,Points:1,Points:2,U:0,U:1,U:2",
        comments="",
    )


def create_estimator(config: ScenarioConfig) -> AirflowEstimator:
    """Create and configure an estimator from a scenario config."""
    estimator = AirflowEstimator.from_mesh(config.mesh)

    wall_names = estimator.match_boundary_names(config.wall_pattern)
    if wall_names:
        estimator.set_no_slip_bc(wall_names)

    outflow_names = estimator.match_boundary_names(config.outflow_pattern)
    if not outflow_names:
        raise ValueError(
            f"No outflow boundaries matched pattern {config.outflow_pattern!r} in {config.mesh.name}."
        )
    estimator.set_zero_pressure_bc(outflow_names)

    estimator.set_regularization(config.regularization)
    estimator.set_weights(
        viscosity=config.viscosity,
        weight_misfit=config.weight_misfit,
        weight_pde_res=config.weight_pde_res,
        weight_reg=config.weight_reg,
        weight_wfns_bc=config.weight_wfns_bc,
    )
    return estimator


def write_outputs(result_dir: Path, velocity: fem.Function, metadata: dict) -> None:
    """Write CSV, quick-look plot, and metadata for one estimation run."""
    from NS_wind_est.visualizer import plot_wind_csv

    estimate_path = result_dir / "wind_estimate.csv"
    plot_path = result_dir / "wind_estimate.png"
    metadata_path = result_dir / "metadata_wind_est.json"

    save_velocity_csv(estimate_path, velocity)
    plot_wind_csv(
        estimate_path,
        output_path=plot_path,
        title="NS wind estimate",
        show=False,
    )
    metadata["wind_estimate_csv"] = str(estimate_path)
    metadata["wind_estimate_png"] = str(plot_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def solve_estimator(
    estimator: AirflowEstimator, config: ScenarioConfig
) -> AirflowResult:
    """Run the solver selected in the scenario config."""
    solver_name = config.solver.strip().upper()
    if solver_name == "SFNS":
        return estimator.solve_SFNS(
            maxit=config.maxit,
            solver_tol=config.solver_tol,
            damping=config.damping,
            verbose=False,
        )
    if solver_name == "WFNS":
        return estimator.solve_WFNS(
            maxit=config.maxit,
            solver_tol=config.solver_tol,
            damping=config.damping,
            verbose=False,
        )
    raise ValueError(f"Unsupported solver {config.solver}, use one of: SFNS, WFNS.")


def run_case(
    config: ScenarioConfig,
    estimator: AirflowEstimator,
    sample_csv: Path,
    result_dir: Path,
    sample_size: int | None,
    verbose: bool,
) -> None:
    """Run one scenario/sample-size/sample-CSV combination."""
    result_dir.mkdir(parents=True, exist_ok=True)

    mapping_info = estimator.set_measurements_from_csv(
        sample_csv,
        count=sample_size,
        noise_std=config.add_gaussian_noise_std,
        max_xy_dist=config.max_xy_dist,
    )
    estimation_start = time.perf_counter()
    result = solve_estimator(estimator, config)
    estimation_runtime_sec = time.perf_counter() - estimation_start
    status = result.status
    status_text = "converged" if status.converged else "reached max iterations"
    change_text = f"{status.final_relative_change:.3e}"
    if verbose:
        print(
            f"NS solver {status_text} "
            f"({sample_size if sample_size is not None else 'all'} samples): "
            f"iterations={status.iterations}/{status.max_iterations}, "
            f"final_relative_change={change_text}, "
            f"solver_tol={config.solver_tol:.3e}"
        )
    u_est = result.velocity

    metadata = {
        "scenario": config.name,
        "sample_name": sample_csv.stem,
        "sample_size": (
            sample_size
            if sample_size is not None
            else int(mapping_info["n_input_samples"])
        ),
        "samples_csv": str(sample_csv),
        "mesh": str(config.mesh),
        "wind_csv": None if config.wind_csv is None else str(config.wind_csv),
        "add_gaussian_noise_std": str(config.add_gaussian_noise_std),
        "solver": config.solver,
        "regularization": config.regularization,
        "maxit": config.maxit,
        "solver_tol": config.solver_tol,
        "damping": config.damping,
        "viscosity": config.viscosity,
        "weight_misfit": config.weight_misfit,
        "weight_pde_res": config.weight_pde_res,
        "weight_reg": config.weight_reg,
        "weight_wfns_bc": config.weight_wfns_bc,
        "n_discretization_points": int(
            u_est.function_space.tabulate_dof_coordinates().shape[0]
        ),
        "estimation_runtime_sec": float(estimation_runtime_sec),
        "solver_used": status.solver,
        "solver_converged": status.converged,
        "solver_iterations": status.iterations,
        "solver_max_iterations": status.max_iterations,
        "solver_final_relative_change": status.final_relative_change,
        **mapping_info,
    }

    write_outputs(result_dir, u_est, metadata)

    if verbose:
        print("---")


def run_scenario(
    scenario: str | Path | ScenarioConfig,
    samples: str | Path | None = None,
    *,
    verbose: bool = False,
) -> ScenarioRunResult:
    """Run one configured scenario for one sample CSV or all scenario samples.

    Parameters
    ----------
    scenario
        Scenario directory, `scenario.yaml` path, or already loaded
        `ScenarioConfig`.
    samples
        Optional path to one sample CSV. If `None`, all `sample_points*.csv`
        files in the scenario sample directory are used.
    verbose
        If `True`, print progress and solver summary lines.

    Returns
    -------
    ScenarioRunResult
        Output root and individual result directories created by the run.
    """
    config = (
        scenario if isinstance(scenario, ScenarioConfig) else ScenarioConfig.load(scenario)
    )
    if samples is not None:
        sample_files = [Path(samples).resolve(strict=True)]
    else:
        sample_files = sorted(config.sample_dir.glob("sample_points*.csv"))
        if not sample_files:
            raise FileNotFoundError(
                f"No sample_points*.csv files found in {config.sample_dir}"
            )

    sample_sizes = config.wind_measurement_counts or (None,)
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_root = config.result_dir / f"{config.solver}-{timestamp}"
    estimator = create_estimator(config)
    result_dirs: list[Path] = []

    for sample_size in sample_sizes:
        for sample_csv in sample_files:
            if verbose:
                label = f" with {sample_size} samples" if sample_size is not None else ""
                print(f"Running {config.solver} for {sample_csv.name}{label}")
            result_dir = output_root
            if len(sample_sizes) > 1 and sample_size is not None:
                result_dir = result_dir / f"{sample_size}samples"
            if len(sample_files) > 1:
                result_dir = result_dir / sample_csv.stem
            run_case(config, estimator, sample_csv, result_dir, sample_size, verbose)
            result_dirs.append(result_dir)

    return ScenarioRunResult(output_root=output_root, result_dirs=tuple(result_dirs))
