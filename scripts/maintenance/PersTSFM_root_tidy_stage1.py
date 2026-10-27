#!/usr/bin/env python3
"""PersTSFM Stage-1 root cleanup: PREVIEW by default; no deletion, no active-script moves.

Commands:
    python scripts/maintenance/PersTSFM_root_tidy_stage1.py
    python scripts/maintenance/PersTSFM_root_tidy_stage1.py --apply
    python scripts/maintenance/PersTSFM_root_tidy_stage1.py --restore archive/manifests/root_tidy_....json

The tool intentionally leaves all existing runnable experiment entry points unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class Move:
    source: str
    destination: str
    sha256: str
    size: int


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for part in iter(lambda: inp.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def in_root(root: Path, rel: str) -> Path:
    """Enforce relative paths, disallow traversal and symlinks in existing components."""
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Unsafe relative path: {rel}')
    out = root / p
    if not out.resolve(strict=False).is_relative_to(root):
        raise ValueError(f'Outside repository root: {rel}')
    return out


def repository_check(root: Path) -> None:
    for required in ['README.md', 'scripts', 'src', 'tests']:
        if not (root / required).exists():
            raise RuntimeError(f'Expected PersTSFM repository; missing {required}')


def target(filename: str) -> str | None:
    if filename.lower().endswith('.zip'):
        return f'archive/packages/{filename}'
    if filename in {'PersTSFM_phase3_3a_fix_rolling_history.py',
                    'PersTSFM_phase2_15_ercot_missing_guard.py'}:
        return f'archive/installers/{filename}'
    if filename == 'README_TIMESTAMP_FIX.md':
        return f'docs/phase3/{filename}'
    if filename == 'PersTSFM_research_status_2026-10-27.md':
        return f'docs/research_snapshots/{filename}'
    if re.search(r'\.before_[\w.-]+\.(?:bak|py)$', filename):
        return f'archive/script_backups/{filename}'
    return None


def code_files(root: Path):
    for directory in ['src', 'scripts', 'tests']:
        base = root / directory
        if not base.is_dir():
            continue
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.py', '.sh', '.toml', '.yaml', '.yml'}:
                if p.stat().st_size <= 750_000:
                    yield p
    for p in root.glob('*.py'):
        if p.is_file() and not p.is_symlink() and p.stat().st_size <= 750_000:
            yield p


def scan_references(root: Path, filenames: list[str]) -> dict[str, list[str]]:
    """Search literal references in runnable code/tests; conservative migration audit."""
    out = {name: [] for name in filenames}
    for f in code_files(root):
        content = f.read_text(encoding='utf-8', errors='replace')
        for name in filenames:
            if f.name == name:  # ignore self-reference in its own implementation
                continue
            if name in content:
                occurrences = [i for i, line in enumerate(content.splitlines(), start=1) if name in line]
                loc = f'{f.relative_to(root)}:' + ','.join(map(str, occurrences[:4]))
                out[name].append(loc)
    return out


def collect_moves(root: Path, references: dict[str, list[str]]) -> tuple[list[Move], list[str]]:
    moves: list[Move] = []
    warnings = []
    for f in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not f.is_file() or f.is_symlink():
            continue
        dest = target(f.name)
        if dest is None:
            continue
        if f.suffix == '.py' and references.get(f.name):
            warnings.append(f'KEEP {f.name}: referenced by ' + '; '.join(references[f.name][:5]))
            continue
        dst = in_root(root, dest)
        if dst.exists():
            # Collision does not mean safe duplicate: never overwrite.
            warnings.append(f'KEEP {f.name}: destination exists: {dest}')
            continue
        moves.append(Move(f.name, dest, sha(f), f.stat().st_size))
    return moves, warnings


def apply_moves(root: Path, moves: list[Move]) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest_rel = f'archive/manifests/root_tidy_{stamp}.json'
    manifest = in_root(root, manifest_rel)
    completed: list[Move] = []
    manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        for m in moves:
            src, dst = in_root(root, m.source), in_root(root, m.destination)
            if not src.is_file() or src.is_symlink() or sha(src) != m.sha256:
                raise RuntimeError(f'Source changed: {src}')
            if dst.exists():
                raise RuntimeError(f'Collision: {dst}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            completed.append(m)
        data = {'version': 1,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'repo_root': str(root),
                'moves': [vars(m) for m in completed],
                'deleted': 0,
                'comment': 'Stage-1 cleanup: no experiment runners, tests, data or results moved.'}
        manifest.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        for m in reversed(completed):
            src, dst = in_root(root, m.source), in_root(root, m.destination)
            if not src.exists() and dst.is_file() and sha(dst) == m.sha256:
                dst.rename(src)
        raise
    return manifest_rel


def restore(root: Path, manifest_rel: str) -> None:
    manifest = in_root(root, manifest_rel)
    if manifest.parent.resolve() != (root / 'archive/manifests').resolve():
        raise ValueError('Restore file must be directly under archive/manifests/')
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    if payload.get('version') != 1:
        raise ValueError('Unknown manifest version')
    moves = [Move(**m) for m in payload['moves']]
    for m in moves:  # Validate every move before restoring anything.
        src, dst = in_root(root, m.source), in_root(root, m.destination)
        if src.exists() or not dst.is_file() or dst.is_symlink() or sha(dst) != m.sha256:
            raise RuntimeError(f'Cannot safely restore {m.destination}; path occupied or bytes changed')
    for m in reversed(moves):
        dst, src = in_root(root, m.destination), in_root(root, m.source)
        dst.rename(src)
    print(f'RESTORED {len(moves)} files; no overwrites; manifest retained')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    act = parser.add_mutually_exclusive_group()
    act.add_argument('--apply', action='store_true')
    act.add_argument('--restore', metavar='ARCHIVE_MANIFEST')
    parser.add_argument('--show-references', action='store_true',
                        help='Print paths in scripts/src/tests that reference root runnable scripts')
    args = parser.parse_args()
    root = args.root.resolve()
    repository_check(root)
    if args.restore:
        restore(root, args.restore)
        return
    candidates = [p.name for p in root.iterdir() if p.is_file() and target(p.name)]
    root_runners = sorted(p.name for p in root.glob('PersTSFM_*.py')
                          if target(p.name) is None)
    refs = scan_references(root, sorted(set(candidates + root_runners)))
    moves, warnings = collect_moves(root, refs)
    print('=' * 80)
    print('PERTSFM ROOT TIDY | ' + ('APPLY' if args.apply else 'DRY RUN'))
    print('=' * 80)
    for m in moves:
        print(f'MOVE {m.source}  ->  {m.destination}')
    for w in warnings:
        print(w)
    print(f'PLANNED {len(moves)} moves; 0 deletions; active runners unchanged')
    print(f'REMAINING ROOT RUNNERS: {len(root_runners)}')
    for filename in root_runners:
        if args.show_references or refs.get(filename):
            matches = refs.get(filename, [])
            print(f'  {filename} | source/test references: {len(matches)}')
            for match in matches[:3]:
                print(f'    {match}')
    print('UNCHANGED: src/, scripts/, tests/, results/, data/, configs/, existing Git changes')
    if args.apply:
        rel = apply_moves(root, moves)
        print(f'APPLIED {len(moves)} moves, deleted 0')
        print(f'MANIFEST {rel}')
        print(f'RESTORE python scripts/maintenance/PersTSFM_root_tidy_stage1.py --restore {rel}')
    else:
        print('Apply only after reviewing: python scripts/maintenance/PersTSFM_root_tidy_stage1.py --apply')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
