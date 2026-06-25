import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WIND_FIELD_COLUMNS = ["Points:0", "Points:1", "Points:2", "U:0", "U:1", "U:2"]
DEFAULT_Z_TOL = 0.05
SINGLE_LAYER_Z_SPAN = 0.1


def load_wind_field_csv(path: str | Path) -> tuple[Path, pd.DataFrame]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Wind CSV not found: {path}")

    rows = pd.read_csv(path)
    missing = [column for column in WIND_FIELD_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}.")

    return path, rows[WIND_FIELD_COLUMNS].copy()


def select_z_slice(rows: pd.DataFrame, z_height: float | None, z_tol: float) -> pd.DataFrame:
    z = rows["Points:2"].to_numpy(dtype=float)
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if z_height is None:
        if z_max - z_min <= SINGLE_LAYER_Z_SPAN:
            return rows
        raise ValueError(
            "Input CSV contains multiple z-levels. Pass --z-height to select a slice. "
            f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
        )

    sliced = rows[np.abs(z - z_height) <= z_tol]
    if sliced.empty:
        raise ValueError(
            f"No rows found within +/- {z_tol} m of z={z_height}. "
            f"Observed z-range is [{z_min:.6g}, {z_max:.6g}]."
        )
    return sliced


def sample_wind_measurements(rows: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if len(rows) < n_samples:
        raise ValueError(
            f"Requested n_samples={n_samples}, but only {len(rows)} valid rows available."
        )

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(rows), size=n_samples, replace=False)
    return rows.iloc[sample_idx].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_csv_samples",
        description="Create sample CSV files from a wind field CSV.",
    )
    parser.add_argument("input_csv", type=str, help="Path to the wind field CSV.")
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        required=True,
        help="Number of measurements per generated sample CSV.",
    )
    parser.add_argument(
        "-s", "--n-sets",
        type=int,
        required=True,
        help="Number of sample CSV files to generate. Seeds 0..n_sets-1 are used.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to the parent directory of the input CSV.",
    )
    parser.add_argument(
        "-z", "--z-height",
        type=float,
        default=None,
        help="Optional z-height of the slice to sample from.",
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=DEFAULT_Z_TOL,
        help=f"Tolerance around --z-height. Defaults to {DEFAULT_Z_TOL}.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wind_csv, rows = load_wind_field_csv(args.input_csv)
    rows = select_z_slice(rows, args.z_height, args.z_tol)

    output_dir = wind_csv.parent if args.output_dir is None else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(args.n_sets):
        sampled = sample_wind_measurements(rows, args.n_samples, seed)
        out_path = output_dir / f"sample_points_n{args.n_samples}_seed{seed}.csv"
        sampled.to_csv(out_path, index=False)
        if args.verbose:
            print(f"Saved {args.n_samples} measurements to {out_path}")


if __name__ == "__main__":
    main()
