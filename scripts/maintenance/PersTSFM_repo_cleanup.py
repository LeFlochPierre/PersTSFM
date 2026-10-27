#!/usr/bin/env python3
"""Reversible, conservative PersTSFM root cleanup.

Default: print proposed moves. --apply: move only known root-level package/docs/
installer/backup artifacts and write a restoration manifest. --restore MANIFEST:
restore files if destination hashes still match and original paths are free.
Never touches data/, results/, src/, scripts/, tests/, configs/ or active phase scripts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def destination(name: str) -> Path | None:
    """Only explicitly classified root-level non-runtime artifacts are moved."""
    if name.lower().endswith('.zip'):
        return Path('archive/packages') / name
    if re.match(r'^install_[\w.-]+\.py$', name, flags=re.I):
        return Path('archive/installers') / name
    if re.search(r'\.before_[\w.-]+\.py$', name, flags=re.I):
        return Path('archive/script_backups') / name
    if re.match(r'^README[_-]PHASE2[_\w.-]*\.md$', name, flags=re.I):
        return Path('docs/phase2') / name
    if re.match(r'^README[_-]PHASE3[_\w.-]*\.md$', name, flags=re.I):
        return Path('docs/phase3') / name
    if re.match(r'^pertsfm_env_before_[\w.-]+\.txt$', name, flags=re.I):
        return Path('docs/environment_snapshots') / name
    if name in {'PersTSFM_project_handover.md',
                'tsfm_personalization_icml_neurips_research_plan.md'}:
        return Path('docs') / name
    return None


def secure_path(root: Path, rel: str) -> Path:
    path = root / rel
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f'Unsafe path outside root: {rel}')
    return path


def repo_status(root: Path) -> None:
    try:
        proc = subprocess.run(['git', 'status', '--short'], cwd=root,
                              capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            lines = proc.stdout.splitlines()
            if lines:
                print(f'NOTE: Git already has {len(lines)} changed/untracked entries. '
                      'Consider a checkpoint commit before --apply.')
    except OSError:
        pass


def plan(root: Path) -> list[dict]:
    planned = []
    for f in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not f.is_file() or f.is_symlink():
            continue
        dest = destination(f.name)
        if dest is None:
            continue
        dst = secure_path(root, str(dest))
        if dst.exists():
            raise RuntimeError(f'Refusing to overwrite destination: {dst}')
        planned.append({'source': f.name, 'destination': dest.as_posix(),
                        'bytes': f.stat().st_size, 'sha256': sha256(f)})
    return planned


def apply(root: Path, records: list[dict]) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest_rel = f'archive/manifests/cleanup_{stamp}.json'
    manifest = secure_path(root, manifest_rel)
    if manifest.exists():
        raise RuntimeError(f'Manifest already exists: {manifest}')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    try:
        for r in records:
            src, dst = secure_path(root, r['source']), secure_path(root, r['destination'])
            if not src.is_file() or src.is_symlink() or sha256(src) != r['sha256']:
                raise RuntimeError(f'Source changed since planning: {src}')
            if dst.exists():
                raise RuntimeError(f'Destination collision: {dst}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            completed.append(r)
    except Exception:
        print('ERROR during move; attempting to reverse completed moves...')
        for r in reversed(completed):
            src, dst = secure_path(root, r['source']), secure_path(root, r['destination'])
            if dst.exists() and not src.exists():
                dst.rename(src)
        raise
    contents = {
        'version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'moves': completed,
        'note': 'No files deleted. Restore using --restore archive/manifests/<filename>.json',
    }
    manifest.write_text(json.dumps(contents, indent=2) + '\n', encoding='utf-8')
    print(f'APPLIED: moved {len(completed)} files, deleted 0')
    print(f'MANIFEST: {manifest_rel}')
    print(f'RESTORE: python PersTSFM_repo_cleanup.py --restore {manifest_rel}')


def restore(root: Path, manifest_arg: str) -> None:
    manifest_path = secure_path(root, manifest_arg)
    expected_dir = secure_path(root, 'archive/manifests').resolve()
    if manifest_path.resolve().parent != expected_dir:
        raise ValueError('Restore manifest must be directly under archive/manifests/')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1:
        raise ValueError('Unknown manifest version')
    moves = manifest['moves']
    for r in moves:  # preflight everything; never partially restore on predictable failures
        src = secure_path(root, r['source'])
        dst = secure_path(root, r['destination'])
        if src.exists():
            raise RuntimeError(f'Original already exists, refusing overwrite: {src}')
        if not dst.is_file() or dst.is_symlink() or sha256(dst) != r['sha256']:
            raise RuntimeError(f'Archived file missing or modified: {dst}')
    for r in reversed(moves):
        src = secure_path(root, r['source'])
        dst = secure_path(root, r['destination'])
        dst.rename(src)
    print(f'RESTORED: {len(moves)} files, no overwrites; manifest retained at {manifest_arg}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--restore', metavar='MANIFEST')
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir() or not (root / 'README.md').is_file():
        ap.error('Expected to run from the PersTSFM repository root (README.md must exist).')
    repo_status(root)
    if args.restore:
        restore(root, args.restore)
        return
    records = plan(root)
    print('PERS-TSFM SAFE ROOT CLEANUP — ' + ('APPLY' if args.apply else 'DRY RUN'))
    for r in records:
        print(f"  {r['source']}  ->  {r['destination']}")
    print(f'PLANNED: {len(records)} files, deleted 0')
    print('UNTOUCHED: every directory, all active phase scripts, results, datasets and tests')
    if args.apply:
        apply(root, records)
    else:
        print('To perform the moves: python PersTSFM_repo_cleanup.py --apply')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
