#!/usr/bin/env python3
"""Conservative, reversible PersTSFM root housekeeping.

Dry run by default.  The only move candidates are untracked root-level
PersTSFM_*.zip and the accidental root-level 'tatus --short' regular file.
Never modifies source, data, results, environments, or hidden Git files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(root: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
    return p.returncode, p.stdout.strip()


def _inside_git_root(root: Path) -> None:
    code, out = _git(root, 'rev-parse', '--show-toplevel')
    if code != 0 or Path(out).resolve() != root.resolve():
        raise RuntimeError(f'Expected a Git project root, got {root}; git says {out}')


def _tracked(root: Path, relative: Path) -> bool:
    return _git(root, 'ls-files', '--error-unmatch', '--', str(relative))[0] == 0


def _destination(relative: Path) -> Path:
    if relative.name == 'tatus --short':
        return Path('archive/triage') / relative.name
    return Path('archive/patches/phase3') / relative.name


def _candidates(root: Path) -> list[Path]:
    root_zips = sorted(p for p in root.glob('PersTSFM_*.zip') if p.parent == root)
    odd = root / 'tatus --short'
    if odd.exists() or odd.is_symlink():
        root_zips.append(odd)
    return root_zips


def _inventory(root: Path) -> dict:
    code, status = _git(root, 'status', '--short')
    if code != 0:
        status = f'git status failed (exit {code})'
    candidates = []
    for p in _candidates(root):
        rel = p.relative_to(root)
        info = {
            'path': str(rel),
            'destination': str(_destination(rel)),
            'tracked': _tracked(root, rel),
            'is_regular_file': p.is_file() and not p.is_symlink(),
            'size_bytes': p.stat().st_size if p.is_file() and not p.is_symlink() else None,
        }
        dest = root / info['destination']
        info['destination_exists'] = dest.exists() or dest.is_symlink()
        info['eligible'] = bool(info['is_regular_file'] and not info['tracked'] and not info['destination_exists'])
        candidates.append(info)
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'repo_root': str(root),
        'git_branch': _git(root, 'branch', '--show-current')[1],
        'git_head': _git(root, 'rev-parse', 'HEAD')[1],
        'git_status_short': status,
        'candidates': candidates,
        'protected_and_unexamined': [
            'src/', 'scripts/phase2/', 'scripts/phase3/', 'configs/', 'tests/',
            'data/', 'results/', 'external/', 'notes/', 'references/',
            'pertsfm_phase2_benchmark/', '.git/', 'src/**/*.bak',
            '__pycache__/', '.pytest_cache/',
        ],
    }


def _apply(root: Path, report: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    manifest_path = root / 'archive' / 'maintenance' / f'housekeeping_{stamp}.json'
    if manifest_path.exists():
        raise RuntimeError(f'Manifest already exists: {manifest_path}')
    moved = []
    for item in report['candidates']:
        if not item['eligible']:
            print('SKIP:', item['path'], '(tracked, non-file, or destination conflict)')
            continue
        src = root / item['path']
        dst = root / item['destination']
        # Check again immediately before each move.
        if not src.is_file() or src.is_symlink() or _tracked(root, Path(item['path'])) or dst.exists() or dst.is_symlink():
            raise RuntimeError(f'Safety condition changed: {src}')
        sha = _sha(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        moved.append({'from': item['path'], 'to': item['destination'], 'sha256': sha, 'size': item['size_bytes']})
        print('MOVED:', item['path'], '->', item['destination'])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'repo_root': str(root),
                'git_head_before': report['git_head'], 'moves': moved,
                'notes': 'All moves are reversible with --restore. No other files were changed.'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('MANIFEST:', manifest_path)
    return manifest_path


def _restore(root: Path, manifest_path: Path) -> None:
    manifest_path = manifest_path.resolve()
    if root not in manifest_path.parents:
        raise RuntimeError('Manifest must be inside this repository')
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    if Path(data['repo_root']).resolve() != root:
        raise RuntimeError('Manifest belongs to a different repository')
    moves = data.get('moves', [])
    # All-or-nothing preflight before restoring anything.
    for move in moves:
        src = root / move['to']
        dst = root / move['from']
        if not src.is_file() or src.is_symlink() or dst.exists() or dst.is_symlink():
            raise RuntimeError(f'Cannot restore {src} -> {dst}; missing source or destination conflict')
        if _sha(src) != move['sha256']:
            raise RuntimeError(f'Checksum mismatch; refusing to restore {src}')
    for move in reversed(moves):
        src = root / move['to']
        dst = root / move['from']
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        print('RESTORED:', move['to'], '->', move['from'])
    print('RESTORE COMPLETE. Original manifest retained for provenance:', manifest_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd(), help='Path to PersTSFM Git root')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true', help='Move eligible untracked files and record checksums')
    group.add_argument('--restore', type=Path, help='Restore a prior move manifest')
    group.add_argument('--json', action='store_true', help='Print dry-run audit as JSON')
    a = p.parse_args()
    root = a.repo.resolve()
    try:
        _inside_git_root(root)
        if a.restore:
            _restore(root, a.restore)
            return 0
        report = _inventory(root)
        if a.json:
            print(json.dumps(report, indent=2))
        else:
            print('PROJECT:', root)
            print('BRANCH:', report['git_branch'], 'HEAD:', report['git_head'][:12])
            print('GIT STATUS:\n' + (report['git_status_short'] or '(clean)'))
            print('\nROOT-LEVEL MOVE PLAN:')
            for x in report['candidates']:
                state = 'READY' if x['eligible'] else 'SKIP (tracked/unsafe/conflict)'
                print(f"  {state}: {x['path']}  ->  {x['destination']}")
            if not report['candidates']:
                print('  No root-level patch ZIPs or suspicious typo file found')
            print('\nNOT TOUCHED:', ', '.join(report['protected_and_unexamined']))
            print('\nDRY RUN ONLY. Pass --apply to perform moves.') if not a.apply else None
        if a.apply:
            _apply(root, report)
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
