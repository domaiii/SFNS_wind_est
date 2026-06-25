"""Loading and validation helpers for YAML scenario cases."""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml

from dataclasses import dataclass, field
from pathlib import Path

SINGLE_LAYER_Z_SPAN = 0.25

def _optional_path(root: Path, value: str | None) -> Path | None:
    return None if value is None else (root / value).resolve()

def _required_path(root: Path, values: dict, fallback: dict, key: str) -> Path:
    value = values.get(key)
    if value is None:
        value = fallback[key]
    return (root / value).resolve()

@dataclass(frozen=True)
class ScenarioConfig:
    """Normalized configuration loaded from a scenario YAML file."""

    name: str
    root: Path
    wind_csv: Path | None
    sample_dir: Path
    result_dir: Path
    add_gaussian_noise_std: float = 0.0
    z_height: float | None = None
    z_tol: float = 0.05
    max_xy_dist: float = 0.2
    wind_measurement_counts: list[int] = field(default_factory=list)
    mesh: Path | None = None
    occupancy_yaml: Path | None = None
    occupancy_image: Path | None = None
    wall_pattern: str = r"wall|obstacle"
    outflow_pattern: str = r"outlet|outflow"
    solver: str = "SFNS"
    regularization: str = "gradient"
    maxit: int = 25
    solver_tol: float = 1e-2
    damping: float | None = None
    viscosity: float = 1e-5
    weight_misfit: float = 1e2
    weight_pde_res: float = 1.0
    weight_reg: float = 1e-2
    weight_wfns_bc: float = 1e4

    @classmethod
    def load(cls, scenario: str | Path) -> "ScenarioConfig":
        """Load ``scenario.yaml`` from a file path or scenario directory."""
        scenario = Path(scenario).resolve()
        if scenario.is_dir():
            scenario = scenario / "scenario.yaml"

        with scenario.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        root = scenario.parent
        info = raw.get("meta", {})
        geometry = raw.get("geometry", {})
        data = raw.get("data", {})
        paths = raw.get("paths", {})
        wind_measurements = raw.get("wind_measurements", {})
        gt_wind_mapping = raw.get("gt_wind_mapping", {})
        solver = raw.get("ns_solver_parameters", {})
        damping = solver.get("damping", cls.damping)
        measurement_counts = wind_measurements.get(
            "sample_sizes",
            wind_measurements.get("counts", raw.get("wind_sample_sizes", [])),
        )

        return cls(
            name=str(info.get("name", raw.get("name", root.name))),
            root=root,
            mesh=_optional_path(root, paths.get("mesh", geometry.get("mesh"))),
            wall_pattern=str(geometry.get("wall_pattern", cls.wall_pattern)),
            outflow_pattern=str(geometry.get("outflow_pattern", cls.outflow_pattern)),
            wind_csv=_optional_path(root, paths.get("wind_csv", data.get("wind_csv"))),
            sample_dir=_required_path(root, paths, data, "sample_dir"),
            result_dir=_required_path(root, paths, data, "result_dir"),
            add_gaussian_noise_std=float(
                wind_measurements.get(
                    "add_gaussian_noise_std",
                    raw.get("measurement_noise", {}).get(
                        "wind_noise_std", cls.add_gaussian_noise_std
                    ),
                )
            ),
            z_height=(
                None
                if gt_wind_mapping.get("z_height") is None
                else float(gt_wind_mapping["z_height"])
            ),
            z_tol=float(gt_wind_mapping.get("z_tol", cls.z_tol)),
            max_xy_dist=float(gt_wind_mapping.get("max_xy_dist", cls.max_xy_dist)),
            wind_measurement_counts=[int(count) for count in measurement_counts],
            solver=str(solver.get("solver", cls.solver)),
            regularization=str(solver.get("regularization", cls.regularization)),
            maxit=int(solver.get("maxit", cls.maxit)),
            solver_tol=float(solver.get("solver_tol", solver.get("tol", cls.solver_tol))),
            damping=None if damping is None else float(damping),
            viscosity=float(solver.get("viscosity", cls.viscosity)),
            weight_misfit=float(solver.get("weight_misfit", cls.weight_misfit)),
            weight_pde_res=float(solver.get("weight_pde_res", cls.weight_pde_res)),
            weight_reg=float(solver.get("weight_reg", cls.weight_reg)),
            weight_wfns_bc=float(solver.get("weight_wfns_bc", cls.weight_wfns_bc))
        )


def infer_z_height(wind_csv: Path, z_height: float | None) -> float:
    """Infer the z-level for single-layer wind CSV data."""
    df = pd.read_csv(wind_csv, usecols=["Points:2"])
    z = df["Points:2"].to_numpy(dtype=float)
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if z_height is not None:
        return z_height
    if z_max - z_min <= SINGLE_LAYER_Z_SPAN:
        return 0.5 * (z_min + z_max)
    raise ValueError(
        "Ground-truth CSV contains multiple z-levels. Pass a z_height in the scenario config. "
        f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
    )
