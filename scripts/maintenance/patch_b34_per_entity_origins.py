#!/usr/bin/env python3
"""Add an opt-in per-entity origin sampler to an installed B3.4 atlas.

Dry-run by default; --apply makes an archived backup. Exact anchors prevent
silently patching an unexpected user-local version of capability_atlas.py.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


def patch(source: str) -> str:
    replacements = [
        (
            '                  coverage_aware: bool = False):',
            '                  coverage_aware: bool = False, per_entity_origins: bool = False):',
        ),
        (
            '''    selection_details = None
    if coverage_aware:
        from .coverage import select_coverage_aware
        selection, selection_details = select_coverage_aware(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins, evaluation_origins=eval_origins, seed=seed)
    else:
''',
            '''    if coverage_aware and per_entity_origins:
        raise ValueError("Choose either --coverage-aware or --per-entity-origins, not both")
    selection_details = None
    if per_entity_origins:
        from .per_entity_origins import select_per_entity_origins
        selection, selection_details = select_per_entity_origins(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins, evaluation_origins=eval_origins,
            seed=seed, resolution_minutes=int(spec.resolution_minutes))
    elif coverage_aware:
        from .coverage import select_coverage_aware
        selection, selection_details = select_coverage_aware(
            protocol, spec.id_col, entities=entities,
            calibration_origins=cal_origins, evaluation_origins=eval_origins, seed=seed)
    else:
''',
        ),
        (
            '''                   selection=("coverage_aware_global_times" if coverage_aware else
                              "shared_origin_times; pooled calibration label maturity enforced"),''',
            '''                   selection=("per_entity_origins_pooled_cutoff" if per_entity_origins else
                              "coverage_aware_global_times" if coverage_aware else
                              "shared_origin_times; pooled calibration label maturity enforced"),''',
        ),
        (
            '''    freeze.add_argument("--coverage-aware", action="store_true",
                        help="Select latest metadata-only time block with enough eligible entities")''',
            '''    freeze.add_argument("--coverage-aware", action="store_true",
                        help="Select latest metadata-only time block with enough eligible entities")
    freeze.add_argument("--per-entity-origins", action="store_true",
                        help="Use entity-specific forecast origins across one pooled calibration/evaluation cutoff")''',
        ),
        (
            '''                      coverage_aware=args.coverage_aware)''',
            '''                      coverage_aware=args.coverage_aware,
                      per_entity_origins=args.per_entity_origins)''',
        ),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"Local B3.4 atlas differs from expected patch anchor: {before[:100]!r}; no changes made")
        source = source.replace(before, after, 1)
    compile(source, 'capability_atlas.py', 'exec')
    return source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    support = root / 'src/budgeted_forecasting/per_entity_origins.py'
    target = root / 'src/budgeted_forecasting/capability_atlas.py'
    if not support.is_file() or not target.is_file():
        raise SystemExit('Required B3.4 file not found; extract the patch in the repository root')
    before = target.read_text(encoding='utf-8')
    if 'per_entity_origins=args.per_entity_origins' in before:
        print('Already patched; no changes')
        return
    after = patch(before)
    if not args.apply:
        print('PREVIEW PASS: B3.4 anchors verified; no files changed')
        return
    archive = root / 'archive/maintenance' / ('b34_entity_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, archive / 'capability_atlas.py')
    target.write_text(after, encoding='utf-8')
    print(f'PATCHED: {target}\nBACKUP: {archive / "capability_atlas.py"}')


if __name__ == '__main__':
    main()
