"""Command-line entry point for configured NS wind scenarios."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NS_wind_est.scenario_runner import run_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ns_wind",
        description="Run the NS wind estimator for one sample CSV or all scenario samples.",
    )
    parser.add_argument("scenario", help="Path to a scenario directory or scenario.yaml.")
    parser.add_argument(
        "-s",
        "--samples",
        help="Path to one sample CSV. If omitted, all scenario samples are used.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print progress details."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_scenario(
        args.scenario,
        samples=args.samples,
        verbose=args.verbose,
    )
    print(f"Saved {result.n_runs} NS runs under: {result.output_root}")


if __name__ == "__main__":
    main()
