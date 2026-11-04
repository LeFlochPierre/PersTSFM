"""Command-line entrypoint for Phase B/C. Invoke from PersTSFM repo root."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .cache import load_cache
from .demo import write_demo
from .feasibility import run_feasibility


def _fresh_path(path: Path, force: bool, directory: bool = False) -> None:
    if path.exists():
        if directory:
            # Never delete a directory or replace unrelated contents.
            outputs = (path / "feasibility.json", path / "decisions.csv")
            if any(p.exists() for p in outputs) and not force:
                raise FileExistsError(f"Existing output files in {path}; choose a new folder or --force")
        elif not force:
            raise FileExistsError(f"Refusing to overwrite {path}; use --force only if intentional")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direction A offline feasibility benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate labelled SYNTHETIC cache")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--force", action="store_true")
    valid = sub.add_parser("validate", help="Validate forecast-cache contract")
    valid.add_argument("--input", required=True, type=Path)
    analysis = sub.add_parser("analyze", help="Run static vs hindsight oracle baseline")
    analysis.add_argument("--input", required=True, type=Path)
    analysis.add_argument("--output-dir", required=True, type=Path)
    analysis.add_argument("--budget-ms", type=float, action="append", required=True)
    analysis.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "demo":
        _fresh_path(args.output, args.force)
        write_demo(args.output)
        print(f"SYNTHETIC cache written: {args.output}")
        return 0
    cases = load_cache(args.input)
    print(f"Validated {len(cases)} cases with {len(cases[0].actions)} actions")
    if args.command == "validate":
        return 0
    report, decisions = run_feasibility(cases, args.budget_ms)
    _fresh_path(args.output_dir, args.force, directory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "feasibility.json"
    out_csv = args.output_dir / "decisions.csv"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(decisions[0])))
        writer.writeheader()
        writer.writerows(asdict(d) for d in decisions)
    print("OFFLINE FEASIBILITY REPORT (no learned or deployable policy)")
    print(f"Reports: {out_json} and {out_csv}")
    for budget in report["budgets"]:
        score = budget["strategies"]
        print(f"  {budget['budget_ms']:g}ms | fixed {score['validation_selected']['macro_mae']:.4f}"
              f" | hindsight oracle {score['hindsight_oracle']['macro_mae']:.4f}"
              f" | gap {budget['oracle_gap_percent_of_fixed']:.1f}%"
              f" | fixed overruns {score['validation_selected']['budget_overruns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
