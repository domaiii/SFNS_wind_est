from pathlib import Path

import pytest

from NS_wind_est.scenario import ScenarioConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = REPO_ROOT / "example_cases/10x6_apartment"


def test_load_example_scenario_contract() -> None:
    config = ScenarioConfig.load(CASE_ROOT)

    assert config.name == "10x6_apartment"
    assert config.mesh == (CASE_ROOT / "geometry/apartment_2d_coarse.msh").resolve()
    assert config.sample_dir == (CASE_ROOT / "samples").resolve()
    assert config.result_dir == (CASE_ROOT / "results").resolve()
    assert config.wind_measurement_counts == [50, 100]
    assert config.solver == "SFNS"
    assert config.regularization == "gradient"
    assert config.damping == pytest.approx(0.5)


def test_load_accepts_scenario_directory_or_yaml_file() -> None:
    from_directory = ScenarioConfig.load(CASE_ROOT)
    from_file = ScenarioConfig.load(CASE_ROOT / "scenario.yaml")

    assert from_directory == from_file


@pytest.mark.parametrize("missing_path", ["mesh", "sample_dir", "result_dir"])
def test_load_rejects_missing_required_path(
    tmp_path: Path, missing_path: str
) -> None:
    paths = {
        "mesh": "geometry/test.msh",
        "sample_dir": "samples",
        "result_dir": "results",
    }
    paths.pop(missing_path)
    yaml_paths = "\n".join(f"  {key}: {value}" for key, value in paths.items())
    (tmp_path / "scenario.yaml").write_text(
        f"paths:\n{yaml_paths}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=missing_path):
        ScenarioConfig.load(tmp_path)
