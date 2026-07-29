"""Command-line entry points."""

from __future__ import annotations

import argparse
import json

from .csvio import read_aep_curve, read_ecology_curve
from .integration import optimize_under_aep_budget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bird-wind-ecology")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tradeoff = subparsers.add_parser("tradeoff", help="Join external AEP and ecology curves")
    tradeoff.add_argument("--aep", required=True, help="External aep_orientation_curve.csv")
    tradeoff.add_argument("--ecology", required=True, help="ecology_orientation_curve.csv")
    tradeoff.add_argument("--budget", type=float, default=0.01, help="Maximum relative AEP loss")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tradeoff":
        result = optimize_under_aep_budget(
            read_aep_curve(args.aep),
            read_ecology_curve(args.ecology),
            args.budget,
        )
        print(json.dumps({
            "farm_id": result.farm_id,
            "budget_fraction": result.budget_fraction,
            "theta_econ_deg": result.theta_econ_deg,
            "theta_eco_deg": result.theta_eco_deg,
            "aep_cost_gwh": result.aep_cost_gwh,
            "relative_risk_reduction": result.relative_risk_reduction,
        }, ensure_ascii=False, indent=2))
        return 0
    return 2

