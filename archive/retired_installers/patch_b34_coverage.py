#!/usr/bin/env python3
"""Non-destructively add --coverage-aware to B3 frozen cohort selection.

Default: dry run. --apply backs up capability_atlas.py under ignored archive/.
Requires previously installed B3.0; fails closed if expected source differs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import shutil


def patch(source: str) -> str:
    replacements = [
        (
            "def freeze_cohort(dataset: str, protocol_split: str, entities: int, cal_origins: int,\n                  eval_origins: int, max_context_hours: float, seed: int, out_dir: Path):",
            "def freeze_cohort(dataset: str, protocol_split: str, entities: int, cal_origins: int,\n                  eval_origins: int, max_context_hours: float, seed: int, out_dir: Path,\n                  coverage_aware: bool = False):",
        ),
        (
            "    selection = select_origins(protocol, spec.id_col, entities=entities,\n                               calibration_origins=cal_origins,\n                               evaluation_origins=eval_origins, seed=seed)\n    loaded = _load_selected(spec, selection)",
            "    selection_details = None\n    if coverage_aware:\n        from .coverage import select_coverage_aware\n        selection, selection_details = select_coverage_aware(\n            protocol, spec.id_col, entities=entities,\n            calibration_origins=cal_origins, evaluation_origins=eval_origins, seed=seed)\n    else:\n        selection = select_origins(protocol, spec.id_col, entities=entities,\n                                   calibration_origins=cal_origins,\n                                   evaluation_origins=eval_origins, seed=seed)\n    loaded = _load_selected(spec, selection)",
        ),
        (
            '                   selection="shared_origin_times; pooled calibration label maturity enforced",\n                   timestamp_note=',
            '                   selection=("coverage_aware_global_times" if coverage_aware else\n                              "shared_origin_times; pooled calibration label maturity enforced"),\n                   selection_details=selection_details,\n                   timestamp_note=',
        ),
        (
            '    freeze.add_argument("--output-dir", type=Path, required=True)',
            '    freeze.add_argument("--output-dir", type=Path, required=True)\n    freeze.add_argument("--coverage-aware", action="store_true",\n                        help="Select latest metadata-only time block with enough eligible entities")',
        ),
        (
            '                      args.evaluation_origins, args.max_context_hours, args.seed, args.output_dir)',
            '                      args.evaluation_origins, args.max_context_hours, args.seed, args.output_dir,\n                      coverage_aware=args.coverage_aware)',
        ),
    ]
    for old,new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"B3 code differs from expected version, no files changed: {old[:80]!r}")
        source=source.replace(old,new,1)
    compile(source, "capability_atlas.py", "exec")
    return source


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    target=root/'src/budgeted_forecasting/capability_atlas.py'
    support=root/'src/budgeted_forecasting/coverage.py'
    if not support.is_file():
        raise SystemExit(f"Missing coverage selector {support}; extract ZIP first")
    original=target.read_text()
    if '--coverage-aware' in original:
        print('Already patched; nothing changed')
        return
    patched=patch(original)
    if not args.apply:
        print('PREVIEW PASS: B3.0 anchors verified; adding optional --coverage-aware; no changes')
        return
    archive=root/'archive/maintenance' / ('b34_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    archive.mkdir(parents=True,exist_ok=False)
    shutil.copy2(target,archive/'capability_atlas.py')
    target.write_text(patched)
    print(f'PATCHED: {target}\nBACKUP: {archive / "capability_atlas.py"}')

if __name__=='__main__':
    main()
