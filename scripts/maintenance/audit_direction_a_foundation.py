#!/usr/bin/env python3
"""Read-only foundation inventory for PersTSFM Direction A; does not load datasets or models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

PACKAGES = (
    'torch', 'chronos-forecasting', 'fev', 'gluonts', 'transformers',
    'peft', 'accelerate', 'numpy', 'pandas', 'pyarrow',
    'statsforecast', 'scikit-learn', 'river', 'mabwiser',
    'vowpalwabbit', 'duckdb', 'optuna', 'pytest',
)
PATHS = (
    'scripts/phase2', 'scripts/phase3', 'scripts/maintenance',
    'src/models', 'src/experiments/adaptation_benchmark',
    'src/analysis', 'src/data', 'src/evaluation',
    'configs/phase2', 'configs/phase3',
    'data', 'external', 'results', 'pertsfm_phase2_benchmark',
)


def git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(['git', *args], cwd=root, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else '<unavailable>'
    except (OSError, subprocess.TimeoutExpired):
        return '<unavailable>'


def inspect(root: Path) -> dict:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    paths = {}
    for rel in PATHS:
        p = root / rel
        paths[rel] = {
            'exists': p.exists(),
            'is_directory': p.is_dir(),
            'top_entries': sorted(x.name for x in p.iterdir())[:40] if p.is_dir() else [],
        }
    return {
        'report_type': 'foundation_read_only',
        'utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'git_branch': git(root, 'branch', '--show-current'),
        'git_head': git(root, 'rev-parse', 'HEAD'),
        'git_status_short': git(root, 'status', '--short'),
        'git_remotes': git(root, 'remote', '-v'),
        'python': sys.version,
        'platform': platform.platform(),
        'environment': {
            'executable': sys.executable,
            'packages': packages,
        },
        'paths': paths,
        'root_patch_zips': sorted(p.name for p in root.glob('PersTSFM_*.zip') if p.is_file()),
        'suspect_root_typo': (root/'tatus --short').exists(),
        'notes': [
            'No model weights or datasets are loaded; package presence does not prove model compatibility.',
            'This audit does not inspect dataset contents, licenses, pretraining overlap or origin viability.',
            'A redacted report is preferable to posting any sensitive paths or contents.'
        ],
    }


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--repo', type=Path, default=Path.cwd())
    a.add_argument('--output', type=Path, default=None, help='Optional JSON destination')
    args = a.parse_args()
    root = args.repo.resolve()
    if not (root/'.git').exists():
        print('ERROR: run from a Git project root, or pass --repo', file=sys.stderr)
        return 2
    report = inspect(root)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        target = args.output.resolve()
        if target.exists():
            print('ERROR: output exists; refusing to overwrite', file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        print(f'AUDIT SAVED: {target}')
    else:
        print(text)
    return 0

if __name__ == '__main__':
    sys.exit(main())
