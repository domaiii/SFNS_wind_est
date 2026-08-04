import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from NS_wind_est.scenario import ScenarioConfig
from NS_wind_est.scenario_runner import run_scenario

from NS_wind_est.airflow_estimator import AirflowEstimator


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = REPO_ROOT / "example_cases/10x6_apartment"


@pytest.mark.smoke
def test_sfns_estimator_smoke() -> None:
    estimator = AirflowEstimator.from_mesh(
        CASE_ROOT / "geometry/apartment_2d_coarse.msh"
    )

    assert "Walls" in estimator.boundary_names
    assert "Outlet" in estimator.boundary_names

    estimator.set_no_slip_bc("Walls")
    estimator.set_zero_pressure_bc("Outlet")
    mapping = estimator.set_measurements_from_csv(
        CASE_ROOT / "samples/sample_points_n400_seed0.csv",
        count=5,
        max_xy_dist=0.2,
    )

    assert mapping["n_input_samples"] == 5
    assert mapping["n_used_samples"] > 0

    result = estimator.solve_SFNS(maxit=1, solver_tol=1e-2)

    assert result.status.solver == "SFNS"
    assert result.status.iterations == 1
    assert np.all(np.isfinite(result.velocity.x.array))
    assert np.all(np.isfinite(result.pressure.x.array))

@pytest.mark.smoke
def test_scenario_workflow_writes_complete_outputs(tmp_path: Path) -> None:
    config = replace(
        ScenarioConfig.load(CASE_ROOT),
        result_dir=tmp_path / "results",
        wind_measurement_counts=[5],
        add_gaussian_noise_std=0.0,
        maxit=1,
    )

    run = run_scenario(
        config,
        samples=CASE_ROOT / "samples/sample_points_n400_seed0.csv",
    )

    assert run.n_runs == 1
    result_dir = run.result_dirs[0]
    estimate_csv = result_dir / "wind_estimate.csv"
    estimate_png = result_dir / "wind_estimate.png"
    metadata_path = result_dir / "metadata_wind_est.json"
    assert estimate_csv.is_file() and estimate_csv.stat().st_size > 0
    assert estimate_png.is_file() and estimate_png.stat().st_size > 0
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["scenario"] == "10x6_apartment"
    assert metadata["sample_size"] == 5
    assert metadata["solver_used"] == "SFNS"
    assert metadata["solver_iterations"] == 1
