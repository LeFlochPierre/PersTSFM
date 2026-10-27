#!/usr/bin/env python3
"""Conservative, reversible migration of legacy PersTSFM root-level entry points.

Dry-run by default. Never edits data/, results/, external/, archive/ (except backups),
tracked changes unrelated to explicitly found path/import references, or tests' logic.
Usage from repo root:
  python scripts/maintenance/migrate_root_entrypoints.py
  python scripts/maintenance/migrate_root_entrypoints.py --apply
  python scripts/maintenance/migrate_root_entrypoints.py --restore archive/manifests/entrypoints_<STAMP>/manifest.json
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

# Explicit allowlist: adding files to a project must not make cleanup move them.
PHASE2 = (
    'scripts/phase2/PersTSFM_phase2_11_temporal_replay.py',
    'scripts/phase2/PersTSFM_phase2_12_joint_temporal_replay.py',
    'scripts/phase2/PersTSFM_phase2_14_cross_task_audit.py',
    'scripts/phase2/PersTSFM_phase2_fev_preflight.py',
    'scripts/phase2/PersTSFM_phase2_temporal_policy_audit.py',
    'scripts/phase2/PersTSFM_temporal_selection_diagnostics.py',
)
PHASE3 = (
    'scripts/phase3/PersTSFM_phase3_0_probe_value.py',
    'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py',
    'scripts/phase3/PersTSFM_phase3_2_accuracy_cost_frontier.py',
    'scripts/phase3/PersTSFM_phase3_2_gpu_cost.py',
    'scripts/phase3/PersTSFM_phase3_3a_chronos_lora.py',
    'scripts/phase3/PersTSFM_phase3_3b_history_audit.py',
)
INSTALLERS = (
    'PersTSFM_phase2_15_ercot_missing_guard.py',
    'PersTSFM_phase3_3a_fix_rolling_history.py',
)
DEST = {
    **{x: 'scripts/phase2/' + x for x in PHASE2},
    **{x: 'scripts/phase3/' + x for x in PHASE3},
    **{x: 'archive/installers/' + x for x in INSTALLERS},
}
ACTIVE = {x: d for x, d in DEST.items() if x not in INSTALLERS}
MOD = {k[:-3]: d[:-3].replace('/', '.') for k, d in ACTIVE.items()}
SKIP_DIRS = {'.git', '.venv', 'venv', 'env', '.pytest_cache', '__pycache__',
             'results', 'data', 'external', 'archive', 'references', 'notes',
             'pertsfm_phase2_benchmark', 'node_modules'}


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_root(root: Path) -> None:
    if not all((root / x).is_dir() for x in ('scripts', 'src', 'tests')) or not (root / 'README.md').is_file():
        raise RuntimeError('Not a PersTSFM repository root (README.md/scripts/src/tests required)')


def source_paths(root: Path):
    # Only executable source/test trees, never historical notes or datasets.
    for p in sorted(root.glob('*.py')):
        if p.is_file() and not p.is_symlink():
            yield p
    for base in ('scripts', 'src', 'tests'):
        for p in sorted((root / base).rglob('*')):
            if (p.is_file() and not p.is_symlink()
                and p.suffix in ('.py', '.sh')
                and not any(part in SKIP_DIRS for part in p.relative_to(root).parts)
                and p.stat().st_size < 2_000_000):
                yield p


def rewrite(text: str, path: Path) -> str:
    # Rewrite direct Python imports before replacing literal file-path strings.
    for old, new in MOD.items():
        # from old import x  -> from scripts.phaseX.old import x
        text = re.sub(r'(?m)^(\s*)from\s+' + re.escape(old) + r'(\s+import\b)',
                      lambda m: m.group(1) + 'from ' + new + m.group(2), text)
        # import old [as alias] -> import scripts.phaseX.old as old/alias
        text = re.sub(r'(?m)^(\s*)import\s+' + re.escape(old) +
                      r'(?:\s+as\s+(\w+))?(\s*(?:#.*)?)$',
                      lambda m: m.group(1) + 'import ' + new + ' as ' +
                      (m.group(2) or old) + m.group(3), text)
        # importlib.import_module('old') or a spec module name string.
        text = re.sub(r'(?P<q>[\"\'])' + re.escape(old) + r'(?P=q)',
                      lambda m: m.group('q') + new + m.group('q'), text)
    for old, new in ACTIVE.items():
        # Only bare file string literals. Avoid changing historical result metadata.
        # Handles Path('old.py'), BASE/'old.py', spec_from_file_location('x', Path(...)).
        text = re.sub(r'(?P<q>[\"\'])(?:\./)?' + re.escape(old) + r'(?P=q)',
                      lambda m: m.group('q') + new + m.group('q'), text)
    return text


def plan(root: Path):
    moves = []
    for filename, relative in DEST.items():
        src, dst = root / filename, root / relative
        if not src.exists():
            continue
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f'Refusing symlink/non-file: {src}')
        if dst.exists():
            raise RuntimeError(f'Destination collision: {dst}; resolve manually')
        moves.append((src, dst))

    # To avoid half-migrated modules, all active legacy files must be present
    # or already in target location. Genuinely absent files are OK.
    moves_by_path = {a: b for a, b in moves}
    edits = []
    for p in source_paths(root):
        original = p.read_bytes()
        try:
            content = original.decode('utf-8')
        except UnicodeError:
            raise RuntimeError(f'Non-UTF-8 source; stop and inspect: {p}')
        updated = rewrite(content, p)
        if updated != content:
            if p.suffix == '.py':
                try: compile(updated, str(moves_by_path.get(p, p)), 'exec')
                except SyntaxError as e: raise RuntimeError(f'Rewrite produced invalid Python {p}: {e}') from e
            edits.append((p, updated.encode('utf-8')))
    return moves, edits


def apply(root: Path, moves, edits):
    if not moves:
        print('No allowlisted root-level files to migrate. Nothing changed.')
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_root = root / 'archive' / 'manifests' / f'entrypoints_{stamp}'
    if backup_root.exists():
        raise RuntimeError('Unexpected backup collision')
    # Backup ALL touched bytes (including files being moved) before mutating.
    paths = set(a for a, _ in moves) | set(p for p, _ in edits)
    records = []
    move_map = {a: b for a, b in moves}
    edits_by_path = {p: b for p, b in edits}
    for p in sorted(paths):
        rel = p.relative_to(root).as_posix()
        original = p.read_bytes()
        post = edits_by_path.get(p, original)
        destination = move_map.get(p, p)
        records.append(dict(source=rel, destination=destination.relative_to(root).as_posix(),
                            before_sha256=digest(original), after_sha256=digest(post),
                            edited=p in edits_by_path, moved=p in move_map))
        bak = backup_root / 'original' / rel
        bak.parent.mkdir(parents=True, exist_ok=True)
        bak.write_bytes(original)
    manifest = dict(version=1, generated_utc=stamp, files=records,
                    note='No files deleted; restore refuses to overwrite post-migration edits.')
    (backup_root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    processed = []
    try:
        for record in records:
            src = root / record['source']
            dst = root / record['destination']
            if record['moved']:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            if record['edited']:
                dst.write_bytes(edits_by_path[src])
            processed.append(record)
        # Validate the source after migration; this executes no experimental code.
        for record in records:
            p = root / record['destination']
            if p.suffix == '.py':
                compile(p.read_bytes(), str(p), 'exec')
            if digest(p.read_bytes()) != record['after_sha256']:
                raise RuntimeError(f'Integrity check failed: {p}')
    except Exception:
        print('ERROR: Migration interrupted; restoring processed files from backups.')
        for record in reversed(processed):
            dest = root / record['destination']
            source = root / record['source']
            backup = backup_root / 'original' / record['source']
            if dest.exists():
                if record['moved']:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if source.exists():
                        raise RuntimeError(f'Automatic rollback conflict: {source}')
                    dest.unlink()
                source.write_bytes(backup.read_bytes())
        raise
    return backup_root / 'manifest.json'


def restore(root: Path, manifest_path: Path):
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to(root / 'archive' / 'manifests'):
        raise RuntimeError('Restore manifest must be inside archive/manifests/')
    obj = json.loads(manifest_path.read_text())
    if obj.get('version') != 1:
        raise RuntimeError('Unsupported restore manifest')
    backup_root = manifest_path.parent / 'original'
    for rec in obj['files']:
        p = root / rec['destination']
        if not p.is_file() or p.is_symlink() or digest(p.read_bytes()) != rec['after_sha256']:
            raise RuntimeError(f'File modified since migration; REFUSE RESTORE: {p}')
        if rec['moved'] and (root / rec['source']).exists():
            raise RuntimeError(f'Original path exists; REFUSE RESTORE: {rec["source"]}')
        b = backup_root / rec['source']
        if digest(b.read_bytes()) != rec['before_sha256']:
            raise RuntimeError(f'Backup checksum mismatch: {b}')
    for rec in reversed(obj['files']):
        dest = root / rec['destination']
        original = root / rec['source']
        if rec['moved']:
            dest.unlink()
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes((backup_root / rec['source']).read_bytes())
    print(f'RESTORED: {len(obj["files"])} original files. Backup manifest retained: {manifest_path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root (default: cwd)')
    ap.add_argument('--apply', action='store_true', help='Execute migration; default is preview')
    ap.add_argument('--restore', type=Path, metavar='MANIFEST', help='Restore prior migration')
    args = ap.parse_args(argv)
    root = args.root.resolve()
    verify_root(root)
    if args.restore:
        restore(root, args.restore if args.restore.is_absolute() else root / args.restore)
        return 0
    moves, edits = plan(root)
    print('PERS-TSFM ROOT SCRIPT MIGRATION — ' + ('APPLY' if args.apply else 'DRY RUN'))
    print('\nSCRIPT MOVES:')
    if not moves: print('  (none)')
    for src, dst in moves:
        print(f'  {src.relative_to(root)} -> {dst.relative_to(root)}')
    print('\nSOURCE/TEST REFERENCE UPDATES:')
    if not edits: print('  (none)')
    for src, _ in edits:
        print(f'  {src.relative_to(root)}' + (' (also moving)' if any(src == a for a, _ in moves) else ''))
    print(f'\nSUMMARY: {len(moves)} moves, {len(edits)} reference edits. No deletions.')
    if args.apply:
        out = apply(root, moves, edits)
        if out:
            print(f'\nAPPLIED. RESTORE MANIFEST: {out.relative_to(root)}')
            print('VERIFY: python -m pytest -q tests/phase2 tests/phase3')
            print('RESTORE: python scripts/maintenance/migrate_root_entrypoints.py --restore ' + str(out.relative_to(root)))
    else:
        print('PREVIEW ONLY. Run again with --apply after reviewing the moves.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
